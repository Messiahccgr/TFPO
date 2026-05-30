from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.utils import setup_logger


logger = setup_logger("teacher")


KEY_TEACHER_PROB = "teacher_prob"
KEY_PREFIX_WEIGHT = "prefix_weight"
KEY_IS_NEGATIVE = "is_negative"
KEY_NEGATIVE_WEIGHT = "negative_weight"
KEY_CANDIDATE_TOKEN_IDS = "candidate_token_ids"
KEY_CANDIDATE_PROBS = "candidate_probs"
KEY_NEGATIVE_TOKEN_ID = "negative_token_id"
KEY_SAMPLE_WEIGHT = "sample_weight"
KEY_EXAMPLE_KIND = "example_kind"

KEY_GRPO_RESPONSE_MASK = "grpo_response_mask"
KEY_GRPO_VALID_MASK = "grpo_valid_mask"
KEY_GRPO_ADVANTAGE = "grpo_advantage"
KEY_GRPO_OLD_LOGPROBS = "grpo_old_logprobs"
KEY_GRPO_HAS_OLD_LOGPROB = "grpo_has_old_logprob"

EXAMPLE_KIND_POSITIVE = "pos"
EXAMPLE_KIND_NEGATIVE = "neg"
EXAMPLE_KIND_GRPO = "grpo"


@dataclass
class TeacherBuildMetrics:
    num_questions: int = 0
    num_rollouts: int = 0
    num_prefixes: int = 0
    num_positive_pairs: int = 0
    num_negative_pairs: int = 0
    num_empty_response: int = 0
    num_duplicate_rollouts: int = 0

    def to_dict(self) -> Dict[str, float]:
        q = max(self.num_questions, 1)
        return {
            "num_questions": float(self.num_questions),
            "num_rollouts_total": float(self.num_rollouts),
            "num_prefixes_total": float(self.num_prefixes),
            "num_positive_pairs_total": float(self.num_positive_pairs),
            "num_negative_pairs_total": float(self.num_negative_pairs),
            "avg_rollouts_per_question": float(self.num_rollouts / q),
            "avg_prefixes_per_question": float(self.num_prefixes / q),
            "avg_positive_pairs_per_question": float(self.num_positive_pairs / q),
            "avg_negative_pairs_per_question": float(self.num_negative_pairs / q),
            "empty_response_count": float(self.num_empty_response),
            "duplicate_rollout_count": float(self.num_duplicate_rollouts),
        }


class ClosedFormTeacherBuilder:
    """Builds TFPO teacher examples from grouped rollouts.

    Faithfully follows Eq. (10)-(13) of the paper:
      - Reliable prefix set S+(x) = {s : K(s) >= m}  (m = min_success_count)
      - Failure frontier F(x) = {(s,a) : K(s) >= k_min, K(s+a)=0, N(s+a) >= n_min}
        (k_min = frontier_min_success_count, n_min = frontier_min_visit_count)
      - Token-level teacher distribution q_hat(a|s) = W(s+a)/W(s) for every observed action.
    """

    def __init__(
        self,
        tokenizer,
        algorithm_cfg: Dict[str, Any],
        max_sequence_length: Optional[int],
    ):
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.eos_token_id = getattr(tokenizer, "eos_token_id", None)

        self.beta = float(algorithm_cfg["beta"])
        self.reward_clip_min = float(algorithm_cfg["reward_clip_min"])
        self.reward_clip_max = float(algorithm_cfg["reward_clip_max"])

        # m: minimum successful rollouts at a prefix for reliable distributional fitting.
        self.min_success_count = int(algorithm_cfg["min_success_count"])
        # k_min: minimum successful rollouts at a frontier parent prefix.
        # Independent from m (paper uses m=24, k_min=8 in Table 4).
        self.frontier_min_success_count = int(
            algorithm_cfg["frontier_min_success_count"]
        )
        # n_min: minimum visit count on the failure-frontier child branch.
        self.frontier_min_visit_count = int(algorithm_cfg["frontier_min_visit_count"])

        self.include_frontier_negative_samples = bool(
            algorithm_cfg["include_frontier_negative_samples"]
        )
        self.negative_weight_mode = str(algorithm_cfg["negative_weight_mode"])
        self.deduplicate_rollouts = bool(
            algorithm_cfg.get("deduplicate_rollouts", True)
        )
        self.append_eos_to_response = bool(
            algorithm_cfg.get("append_eos_to_response", True)
        )

    def reward_to_weight(self, reward: float) -> float:
        clipped = float(np.clip(reward, self.reward_clip_min, self.reward_clip_max))
        # exp() bounds: prevent fp64 over/underflow when beta is small or reward
        # bounds are widened. Has no effect under the default (beta=0.1, R in [0,1]).
        exponent = float(np.clip(clipped / self.beta, -50.0, 125.0))
        return float(np.exp(exponent))

    @staticmethod
    def reward_is_success(reward: float) -> bool:
        return float(reward) > 0.0

    @staticmethod
    def rollout_is_success(rollout: Dict[str, Any]) -> bool:
        if "answer_correct" in rollout:
            return bool(rollout["answer_correct"])
        return ClosedFormTeacherBuilder.reward_is_success(
            float(rollout.get("reward", 0.0))
        )

    def _tokenize_trajectory(
        self, query_text: str, response_text: str
    ) -> Tuple[List[int], List[int]]:
        query_ids = self.tokenizer(query_text, add_special_tokens=False).input_ids
        full_ids = self.tokenizer(
            query_text + response_text, add_special_tokens=False
        ).input_ids

        if len(full_ids) >= len(query_ids):
            response_ids = full_ids[len(query_ids) :]
        else:
            response_ids = []

        if len(response_ids) == 0 and len(response_text) > 0:
            response_ids = self.tokenizer(
                response_text, add_special_tokens=False
            ).input_ids

        if (
            self.append_eos_to_response
            and self.eos_token_id is not None
            and (len(response_ids) == 0 or response_ids[-1] != int(self.eos_token_id))
        ):
            response_ids = list(response_ids) + [int(self.eos_token_id)]

        return query_ids, response_ids

    @staticmethod
    def _renormalize(
        action_prob_pairs: List[Tuple[int, float]]
    ) -> List[Tuple[int, float]]:
        if len(action_prob_pairs) == 0:
            return []
        prob_sum = sum(prob for _, prob in action_prob_pairs)
        if prob_sum <= 0:
            return []
        return [(tok, prob / prob_sum) for tok, prob in action_prob_pairs]

    def _collect_prefix_tree_stats(
        self, trajectories: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        node_total_weight: Dict[Tuple[int, ...], float] = defaultdict(float)
        node_visit_count: Dict[Tuple[int, ...], int] = defaultdict(int)
        node_success_count: Dict[Tuple[int, ...], int] = defaultdict(int)
        edge_weight: Dict[Tuple[int, ...], Dict[int, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        edge_visit_count: Dict[Tuple[int, ...], Dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        for traj in trajectories:
            response_tokens = traj["response_token_ids"]
            if len(response_tokens) == 0:
                continue

            weight = traj["weight"]
            is_success = bool(traj["is_success"])

            for t in range(len(response_tokens) + 1):
                prefix = tuple(response_tokens[:t])
                node_total_weight[prefix] += weight
                node_visit_count[prefix] += 1
                if is_success:
                    node_success_count[prefix] += 1

            for t, action_token in enumerate(response_tokens):
                prefix = tuple(response_tokens[:t])
                action_token = int(action_token)
                edge_weight[prefix][action_token] += weight
                edge_visit_count[prefix][action_token] += 1

        return {
            "node_total_weight": node_total_weight,
            "node_visit_count": node_visit_count,
            "node_success_count": node_success_count,
            "edge_weight": edge_weight,
            "edge_visit_count": edge_visit_count,
        }

    def _build_pairs_for_question(
        self,
        question_idx: Any,
        query_token_ids: List[int],
        trajectories: List[Dict[str, Any]],
    ) -> Tuple[
        List[Dict[str, Any]],
        Dict[str, float],
        Dict[Tuple[int, ...], int],
        Dict[Tuple[Tuple[int, ...], int], Dict[str, int]],
    ]:
        """Returns (examples, metrics, reliable_prefix_K, frontier_edges).

        reliable_prefix_K maps s -> K(s) for every s in S+(x).
        frontier_edges maps (s, a) -> stats for every (s,a) in F(x).
        """
        trie = self._collect_prefix_tree_stats(trajectories)
        node_total_weight = trie["node_total_weight"]
        node_visit_count = trie["node_visit_count"]
        node_success_count = trie["node_success_count"]
        edge_weight = trie["edge_weight"]
        edge_visit_count = trie["edge_visit_count"]

        # S+(x) = {s : K(s) >= m}.
        reliable_prefix_K: Dict[Tuple[int, ...], int] = {
            s: int(node_success_count[s])
            for s in edge_weight.keys()
            if int(node_success_count[s]) >= self.min_success_count
        }

        positive_examples: List[Dict[str, Any]] = []
        for prefix in reliable_prefix_K.keys():
            total_weight = float(node_total_weight[prefix])
            if total_weight <= 0:
                continue

            action_prob_pairs = [
                (int(tok), weight / total_weight)
                for tok, weight in edge_weight[prefix].items()
            ]
            action_prob_pairs = self._renormalize(action_prob_pairs)
            if len(action_prob_pairs) == 0:
                continue

            state_query_token_ids = query_token_ids + list(prefix)
            if (
                self.max_sequence_length is not None
                and len(state_query_token_ids) + 1 > self.max_sequence_length
            ):
                continue

            positive_examples.append(
                {
                    "question_idx": question_idx,
                    "query_token_ids": state_query_token_ids,
                    KEY_EXAMPLE_KIND: EXAMPLE_KIND_POSITIVE,
                    KEY_CANDIDATE_TOKEN_IDS: [int(a) for a, _ in action_prob_pairs],
                    KEY_CANDIDATE_PROBS: [float(p) for _, p in action_prob_pairs],
                    KEY_PREFIX_WEIGHT: 1.0,
                    KEY_IS_NEGATIVE: 0.0,
                    KEY_NEGATIVE_WEIGHT: 1.0,
                    KEY_NEGATIVE_TOKEN_ID: -100,
                }
            )

        # F(x) = {(s,a) : K(s) >= k_min, K(s+a) = 0, N(s+a) >= n_min}.
        frontier_edges: Dict[Tuple[Tuple[int, ...], int], Dict[str, int]] = {}
        negative_examples: List[Dict[str, Any]] = []
        if self.include_frontier_negative_samples:
            for prefix, action_visit_map in edge_visit_count.items():
                if int(node_success_count[prefix]) < self.frontier_min_success_count:
                    continue
                state_query_token_ids = query_token_ids + list(prefix)
                if (
                    self.max_sequence_length is not None
                    and len(state_query_token_ids) + 1 > self.max_sequence_length
                ):
                    continue

                for action_token, visit_count in action_visit_map.items():
                    child_prefix = prefix + (int(action_token),)
                    if int(node_success_count[child_prefix]) != 0:
                        continue
                    child_visit_count = int(node_visit_count[child_prefix])
                    if child_visit_count < self.frontier_min_visit_count:
                        continue

                    edge_key = (prefix, int(action_token))
                    frontier_edges[edge_key] = {
                        "parent_K": int(node_success_count[prefix]),
                        "parent_N": int(node_visit_count[prefix]),
                        "child_N": child_visit_count,
                        "edge_N": int(visit_count),
                    }

                    neg_weight = (
                        float(visit_count)
                        if self.negative_weight_mode == "visit_count"
                        else 1.0
                    )
                    negative_examples.append(
                        {
                            "question_idx": question_idx,
                            "query_token_ids": state_query_token_ids,
                            KEY_EXAMPLE_KIND: EXAMPLE_KIND_NEGATIVE,
                            KEY_CANDIDATE_TOKEN_IDS: [],
                            KEY_CANDIDATE_PROBS: [],
                            KEY_PREFIX_WEIGHT: 0.0,
                            KEY_IS_NEGATIVE: 1.0,
                            KEY_NEGATIVE_WEIGHT: neg_weight,
                            KEY_NEGATIVE_TOKEN_ID: int(action_token),
                        }
                    )

        pos_count = max(len(positive_examples), 1)
        neg_count = max(len(negative_examples), 1)
        for ex in positive_examples:
            ex[KEY_SAMPLE_WEIGHT] = 1.0 / float(pos_count)
        for ex in negative_examples:
            ex[KEY_SAMPLE_WEIGHT] = 1.0 / float(neg_count)

        examples = positive_examples + negative_examples
        metrics = {
            "num_prefixes": float(len(reliable_prefix_K)),
            "num_positive_pairs": float(len(positive_examples)),
            "num_negative_pairs": float(len(negative_examples)),
        }
        return examples, metrics, reliable_prefix_K, frontier_edges

    def _build_bad_frontier_edges_for_question(
        self,
        question_idx: Any,
        query_token_ids: List[int],
        trajectories: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """Standalone frontier extraction for offline statistics (not used in training loss)."""
        trie = self._collect_prefix_tree_stats(trajectories)
        node_visit_count = trie["node_visit_count"]
        node_success_count = trie["node_success_count"]
        edge_visit_count = trie["edge_visit_count"]

        bad_edges: List[Dict[str, Any]] = []
        for prefix, action_visit_map in edge_visit_count.items():
            parent_success_count = int(node_success_count[prefix])
            if parent_success_count < self.frontier_min_success_count:
                continue

            state_query_token_ids = query_token_ids + list(prefix)
            if (
                self.max_sequence_length is not None
                and len(state_query_token_ids) + 1 > self.max_sequence_length
            ):
                continue

            for action_token, visit_count in action_visit_map.items():
                child_prefix = prefix + (int(action_token),)
                if int(node_success_count[child_prefix]) != 0:
                    continue
                child_visit_count = int(node_visit_count[child_prefix])
                if child_visit_count < self.frontier_min_visit_count:
                    continue

                bad_edges.append(
                    {
                        "question_idx": question_idx,
                        "query_token_ids": state_query_token_ids,
                        KEY_NEGATIVE_TOKEN_ID: int(action_token),
                        "parent_prefix_len": int(len(prefix)),
                        "parent_visit_count": int(node_visit_count[prefix]),
                        "parent_success_count": parent_success_count,
                        "child_visit_count": child_visit_count,
                        "edge_visit_count": int(visit_count),
                    }
                )

        return bad_edges, {"num_bad_edges": float(len(bad_edges))}

    def _prepare_trajectories(
        self,
        query_text: str,
        rollouts: List[Dict[str, Any]],
        metrics: TeacherBuildMetrics,
    ) -> List[Dict[str, Any]]:
        trajectories: List[Dict[str, Any]] = []
        seen_rollout_keys = set()
        for rollout in rollouts:
            response_text = rollout["response_text"]
            _, response_token_ids = self._tokenize_trajectory(
                query_text=query_text,
                response_text=response_text,
            )

            if len(response_token_ids) == 0:
                metrics.num_empty_response += 1
                continue

            rollout_key = tuple(int(tok) for tok in response_token_ids)
            if self.deduplicate_rollouts and rollout_key in seen_rollout_keys:
                metrics.num_duplicate_rollouts += 1
                continue
            seen_rollout_keys.add(rollout_key)

            reward = float(rollout["reward"])
            token_logprobs = self._align_token_logprobs(
                rollout.get("token_logprobs"),
                expected_len=len(response_token_ids),
            )
            trajectories.append(
                {
                    "query_text": query_text,
                    "response_token_ids": response_token_ids,
                    "reward": reward,
                    "is_success": self.rollout_is_success(rollout),
                    "weight": self.reward_to_weight(reward),
                    "token_logprobs": token_logprobs,
                }
            )
            metrics.num_rollouts += 1
        return trajectories

    @staticmethod
    def _align_token_logprobs(
        token_logprobs: Optional[List[float]],
        expected_len: int,
    ) -> Optional[List[float]]:
        """Align vLLM-returned per-token logprobs with HF-tokenized response.

        vLLM token strings and HF token ids can disagree by ±1-2 around EOS or
        special-token boundaries. Small mismatches are silently trimmed/padded;
        larger ones drop the field so GRPO falls back to ratio=1 for that rollout.
        """
        if token_logprobs is None:
            return None
        diff = len(token_logprobs) - expected_len
        if diff == 0:
            return [float(x) for x in token_logprobs]
        if -2 <= diff <= 2:
            if diff > 0:
                return [float(x) for x in token_logprobs[:expected_len]]
            return [float(x) for x in token_logprobs] + [0.0] * (-diff)
        logger.warning(
            "token_logprobs length %d mismatches response_token_ids %d (>2); "
            "dropping logprobs for this rollout.",
            len(token_logprobs),
            expected_len,
        )
        return None

    def build_for_batch(
        self, batch_rollouts: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float], Dict[Any, Dict[str, Any]]]:
        """Returns (examples, metrics, per_question_aux).

        per_question_aux[qid] = {
            "query_token_ids": [...],
            "trajectories": [...],
            "reliable_prefix_K": {s: K(s)},
            "frontier_edges": {(s,a): {...}},
        }
        Downstream GRPO builder uses per_question_aux to compute valid_for_grpo masks
        without re-tokenizing or re-building the trie.
        """
        all_examples: List[Dict[str, Any]] = []
        metrics = TeacherBuildMetrics(num_questions=len(batch_rollouts))
        per_question_aux: Dict[Any, Dict[str, Any]] = {}

        for sample in batch_rollouts:
            question_idx = sample["question_idx"]
            query_text = sample["query_text"]
            rollouts = sample["rollouts"]
            query_token_ids = self.tokenizer(
                query_text, add_special_tokens=False
            ).input_ids

            trajectories = self._prepare_trajectories(
                query_text=query_text,
                rollouts=rollouts,
                metrics=metrics,
            )

            if len(trajectories) == 0:
                continue

            examples, q_metrics, reliable_prefix_K, frontier_edges = (
                self._build_pairs_for_question(
                    question_idx=question_idx,
                    query_token_ids=query_token_ids,
                    trajectories=trajectories,
                )
            )
            all_examples.extend(examples)
            metrics.num_prefixes += int(q_metrics["num_prefixes"])
            metrics.num_positive_pairs += int(q_metrics["num_positive_pairs"])
            metrics.num_negative_pairs += int(q_metrics["num_negative_pairs"])

            per_question_aux[question_idx] = {
                "query_token_ids": query_token_ids,
                "trajectories": trajectories,
                "reliable_prefix_K": reliable_prefix_K,
                "frontier_edges": frontier_edges,
            }

        return all_examples, metrics.to_dict(), per_question_aux

    def build_grpo_examples(
        self,
        batch_rollouts: List[Dict[str, Any]],
        per_question_aux: Dict[Any, Dict[str, Any]],
        grpo_skip_after_frontier: bool = False,
        advantage_eps: float = 1e-6,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """Builds per-rollout GRPO examples for tokens not covered by TFPO.

        valid_mask[i] = True iff token input_ids[i] is a response token AND
        its prefix s_t = response[:t] is NOT in S+(x) AND (s_t, a_{t+1}) is NOT
        a failure-frontier edge. When ``grpo_skip_after_frontier`` is True, all
        response tokens after the first frontier edge are also masked out.

        Advantage is computed group-relative within each question. Trajectories
        whose token-logprobs are missing/misaligned fall back to ratio=1
        (REINFORCE-style step); a per-example flag records this.
        """
        examples: List[Dict[str, Any]] = []
        total_resp_tokens = 0
        total_valid_tokens = 0
        num_no_logprob = 0

        for sample in batch_rollouts:
            qid = sample["question_idx"]
            aux = per_question_aux.get(qid)
            if aux is None:
                continue
            query_token_ids = aux["query_token_ids"]
            trajectories = aux["trajectories"]
            reliable_K = aux["reliable_prefix_K"]
            frontier_edges = aux["frontier_edges"]
            if len(trajectories) <= 1:
                continue

            rewards = [float(t["reward"]) for t in trajectories]
            mean_r = sum(rewards) / len(rewards)
            var_r = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
            std_r = max(var_r, advantage_eps) ** 0.5

            for traj, raw_reward in zip(trajectories, rewards):
                response_token_ids = traj["response_token_ids"]
                if len(response_token_ids) == 0:
                    continue

                full_ids = list(query_token_ids) + list(response_token_ids)
                if (
                    self.max_sequence_length is not None
                    and len(full_ids) > self.max_sequence_length
                ):
                    full_ids = full_ids[: self.max_sequence_length]
                response_start = len(query_token_ids)
                response_len = len(full_ids) - response_start
                if response_len <= 0:
                    continue

                advantage = (raw_reward - mean_r) / std_r

                response_mask = [0] * response_start + [1] * response_len

                valid_mask = [0] * len(full_ids)
                hit_frontier = False
                for t in range(response_len):
                    action = int(full_ids[response_start + t])
                    prefix = tuple(response_token_ids[:t])

                    edge_key = (prefix, action)
                    is_frontier_edge = edge_key in frontier_edges

                    if grpo_skip_after_frontier and hit_frontier:
                        continue
                    if is_frontier_edge:
                        hit_frontier = True

                    in_reliable = prefix in reliable_K
                    if not in_reliable and not is_frontier_edge:
                        valid_mask[response_start + t] = 1

                old_logprobs = [0.0] * len(full_ids)
                traj_logprobs = traj.get("token_logprobs")
                has_old = (
                    traj_logprobs is not None
                    and len(traj_logprobs) >= response_len
                )
                if has_old:
                    for t in range(response_len):
                        old_logprobs[response_start + t] = float(traj_logprobs[t])
                else:
                    num_no_logprob += 1

                examples.append(
                    {
                        "question_idx": qid,
                        "query_token_ids": full_ids,
                        KEY_EXAMPLE_KIND: EXAMPLE_KIND_GRPO,
                        KEY_CANDIDATE_TOKEN_IDS: [],
                        KEY_CANDIDATE_PROBS: [],
                        KEY_PREFIX_WEIGHT: 0.0,
                        KEY_IS_NEGATIVE: 0.0,
                        KEY_NEGATIVE_WEIGHT: 0.0,
                        KEY_NEGATIVE_TOKEN_ID: -100,
                        KEY_SAMPLE_WEIGHT: 1.0,
                        KEY_GRPO_RESPONSE_MASK: response_mask,
                        KEY_GRPO_VALID_MASK: valid_mask,
                        KEY_GRPO_ADVANTAGE: float(advantage),
                        KEY_GRPO_OLD_LOGPROBS: old_logprobs,
                        KEY_GRPO_HAS_OLD_LOGPROB: bool(has_old),
                    }
                )

                total_resp_tokens += response_len
                total_valid_tokens += sum(valid_mask[response_start:])

        metrics = {
            "num_grpo_examples": float(len(examples)),
            "grpo_total_response_tokens": float(total_resp_tokens),
            "grpo_valid_tokens": float(total_valid_tokens),
            "grpo_valid_token_frac": float(
                total_valid_tokens / max(total_resp_tokens, 1)
            ),
            "grpo_examples_without_logprob": float(num_no_logprob),
        }
        return examples, metrics

    def build_bad_frontier_edges(
        self, batch_rollouts: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        all_bad_edges: List[Dict[str, Any]] = []
        num_questions = len(batch_rollouts)
        num_rollouts = 0
        num_empty_response = 0
        num_duplicate_rollouts = 0

        scratch_metrics = TeacherBuildMetrics()
        for sample in batch_rollouts:
            question_idx = sample["question_idx"]
            query_text = sample["query_text"]
            rollouts = sample["rollouts"]
            query_token_ids = self.tokenizer(
                query_text, add_special_tokens=False
            ).input_ids

            trajectories = self._prepare_trajectories(
                query_text=query_text,
                rollouts=rollouts,
                metrics=scratch_metrics,
            )
            num_rollouts += scratch_metrics.num_rollouts
            num_empty_response += scratch_metrics.num_empty_response
            num_duplicate_rollouts += scratch_metrics.num_duplicate_rollouts
            scratch_metrics = TeacherBuildMetrics()

            if len(trajectories) == 0:
                continue

            bad_edges, _ = self._build_bad_frontier_edges_for_question(
                question_idx=question_idx,
                query_token_ids=query_token_ids,
                trajectories=trajectories,
            )
            all_bad_edges.extend(bad_edges)

        return all_bad_edges, {
            "num_questions": float(num_questions),
            "num_rollouts_total": float(num_rollouts),
            "num_bad_edges": float(len(all_bad_edges)),
            "avg_bad_edges_per_question": float(
                len(all_bad_edges) / max(num_questions, 1)
            ),
            "empty_response_count": float(num_empty_response),
            "duplicate_rollout_count": float(num_duplicate_rollouts),
        }

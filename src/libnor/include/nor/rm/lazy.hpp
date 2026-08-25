
#ifndef NOR_RM_LAZY_HPP
#define NOR_RM_LAZY_HPP

#include <vector>

namespace nor::rm::lazy {

/// Per-infoset bookkeeping of the Lazy-CFR segmentation (Zhou et al., "Lazy-CFR: fast and
/// near-optimal regret minimization for extensive games", ICLR 2020, arXiv:1810.04433).
///
/// Time is segmented PER INFOSET: inside a segment the infoset's recommendation (strategy) is
/// FROZEN; counterfactual regret increments and own-reach-weighted average-strategy mass
/// arriving while the segment is open are buffered here instead of being applied eagerly.
/// When the accumulated OPPONENT reach pi_{-i}(I) since the last refresh exhausts the budget
/// B the segment closes: both buffers fold into the tables at once -- exact under the frozen
/// strategy, because sum_t pi_i^t(I) * sigma^t(I, a) collapses to
/// (sum_t pi_i^t(I)) * sigma(I, a) when sigma is constant across the folded iterations --
/// and only then is the regret-matching recommendation recomputed.
struct SegmentState {
   /// buffered instantaneous counterfactual regret increments r~(I, a) of the OPEN segment;
   /// entry i is index-aligned with the owning infostate's action registry
   std::vector< double > regret_buffer;
   /// accumulated counterfactual (opponent) reach pi_{-i}(h) over the open segment's visits;
   /// the segment-closing trigger against the configured budget B
   double pending_cf_reach = 0.;
   /// accumulated OWN reach pi_i(h) over the open segment's visits; applied as one deferred
   /// average-strategy mass at fold time
   double pending_player_reach = 0.;
   /// set when the segment closed during the latest traversal and its buffers were folded;
   /// consumed (and cleared) by the end-of-iteration sweep, which then recomputes the
   /// recommendation. While false the sweep leaves the strategy frozen.
   bool refresh_pending = false;
};

}  // namespace nor::rm::lazy

#endif  // NOR_RM_LAZY_HPP


#ifndef NOR_RM_TEAM_TEAM_BELIEF_DAG_HPP
#define NOR_RM_TEAM_TEAM_BELIEF_DAG_HPP

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "common/common.hpp"
#include "nor/at_runtime.hpp"
#include "nor/concepts.hpp"
#include "nor/game_defs.hpp"
#include "nor/type_defs.hpp"
#include "nor/utils/utils.hpp"

namespace nor::rm::team {

/// dense ids shared by the TB-DAG builder and its consumers
using NodeId = std::size_t;
using BeliefId = std::size_t;
using InactiveId = std::size_t;

inline constexpr std::size_t dag_npos = std::numeric_limits< std::size_t >::max();

/**
 * @brief Team Belief DAG form of Zhang, Farina & Sandholm (ICML 2022, arXiv:2202.00789).
 *
 * This class constructs the team belief DAG (TB-DAG) of the TEAM DECISION PROBLEM induced by
 * fixing a cooperating coalition ('members') inside an adversarial extensive-form game given as
 * a FOSG environment. The construction follows Section 4 of the paper verbatim:
 *
 * - The TEAM decision problem H is the full game world-tree restricted attention of the
 *   coalition: a world-tree node h is ACTIVE iff one of the team members decides there; all
 *   chance/opponent nodes are INACTIVE. Members track their FOSG infostates along the
 *   enumeration, inducing the team's information partition.
 *
 * - The CONNECTIVITY GRAPH G (Definition 4.1) connects h,h' of the same tree layer whenever
 *   some team infoset I exists with h precursor-of I and h' precursor-of I -- equivalently,
 *   whenever their ancestor-or-self team-infoset sets intersect. This encodes precisely what
 *   information is NOT yet common knowledge inside the team.
 *
 * - BELIEFS (the active nodes of the DAG) are sets of team worlds; PRESCRIPTIONS (the actions
 *   at a belief) are cartesian tuples assigning one legal action to every team infoset that
 *   intersects the belief. INACTIVE nodes collect all prescription-consistent successors and
 *   decompose them into PUBLIC OBSERVATIONS: the connected components of G over their worlds
 *   (step 2/3 of the paper's recursive definition). Nodes sharing an identical world-set are
 *   MERGED, producing a perfect-information DAG that is strategically equivalent to the team
 *   decision problem (Theorem 4.2) while collapsing the exponentially larger team product
 *   space through common-information equivalence classes.
 *
 * Construction runs ONE depth-first enumeration of the underlying game tree. Chance
 * probabilities are folded into per-terminal weights so consumers obtain, per leaf z, the
 * scalar p_c(z) together with the payoff of every root participant. Realization flows of a
 * behavioral strategy on the DAG induce a well-defined correlation plan over the original
 * game tree; see rm::team::AdversarialTeamDagCfr for the DAG-CFR learner built on top.
 * A later per-member behavioral-policy view is only the marginal projection of that correlated
 * plan; independently sampling those marginals is not generally payoff-equivalent.
 *
 * DEVIATIONS FROM THE PAPER: none in the construction itself. Enumeration order determines
 * only internal ids (labels are canonicalized via sorted world-id vectors), so the produced
 * DAG shape is order-independent up to relabeling.
 */
template < typename Env >
class TeamBeliefDAG {
  public:
   using env_type = Env;
   using world_state_type = auto_world_state_type< Env >;
   using info_state_type = auto_info_state_type< Env >;
   using action_type = auto_action_type< Env >;
   using chance_outcome_type = auto_chance_outcome_type< Env >;

   static_assert(
      concepts::fosg< Env >,
      "TeamBeliefDAG requires the environment to fulfill the fosg concept"
   );

   /// hard default safety valve against accidental exponential blowups
   static constexpr size_t k_default_max_dag_nodes = 10'000'000;

   struct Config {
      /// the cooperating team members (>= 1 seat). All remaining participants of the root
      /// roster act as the adversarial block (or are inert scenery for degenerate configs).
      std::vector< Player > members{};
      /// upper bound on the number of DAG nodes constructed; a blowup beyond this throws.
      size_t max_dag_nodes = k_default_max_dag_nodes;
   };

   ///////////////////////
   /// public structure //
   ///////////////////////

   using NodeId = size_t;  //< world-tree node id of the enumerated team decision problem
   using BeliefId = size_t;
   using InactiveId = size_t;

   static constexpr size_t npos = std::numeric_limits< size_t >::max();

   /// one enumerated world-tree node of the team decision problem (exposed for testing and
   /// introspection; the TB-DAG proper lives in 'Belief'/'InactiveNode')
   struct WorldNode {
      size_t depth = 0;
      bool terminal = false;
      /// true iff a TEAM MEMBER decides at this world node
      bool team_decision = false;
      /// deciding participant (decision nodes) or active agent (chance / adversary nodes)
      Player owner = Player::unknown;
      /// the owner's member-local infoset id iff team_decision, npos otherwise
      size_t infoset_local = npos;
      /// the GLOBAL team-infoset id iff team_decision, npos otherwise
      size_t infoset_global = npos;
      /// sorted ascending set of team-infoset global ids reachable in this node's subtree
      /// (including itself) -- the connectivity-graph encoding of Definition 4.1
      std::vector< size_t > anc_infosets;
      /// successor world nodes (all outcomes at inactive nodes, one-per-action at decisions,
      /// aligned with the environment's canonical legal-action / chance-outcome order)
      std::vector< NodeId > children;
      /// product of chance probabilities from the root down to this node
      double chance_weight = 1.;
      /// participant-indexed rewards; only meaningful at terminal nodes (aligned with
      /// 'roster()', i.e. the root roster without chance)
      std::vector< double > payoffs;
   };

   /// one team infoset slot inside a belief: a member infoset intersecting the belief together
   /// with its canonical legal-action list
   struct BeliefSlot {
      size_t member_idx = 0;  //< index into 'members()' / member registries
      size_t infoset_global = 0;
      std::vector< action_type > actions;
   };

   /// a prescription: an assignment of one action index (into the corresponding slot's
   /// canonical action list) for every slot of the owning belief -- the joint instruction the
   /// team controller would broadcast at the associated common-information state
   struct Prescription {
      std::vector< size_t > slot_action_idx;
   };

   /// a belief: the DAG's active node, labeled with a set of team worlds
   struct Belief {
      /// ascending world-node ids forming the belief label (also the memo key)
      std::vector< NodeId > worlds;
      /// team infosets intersecting the belief, in first-discovery (ascending world) order
      std::vector< BeliefSlot > slots;
      std::vector< Prescription > prescriptions;
      /// inactive child reached by prescription i (parallel to 'prescriptions')
      std::vector< InactiveId > prescription_children;
      /// inactive parents of this belief (uniquely structured: <= parents of prior inactive)
      std::vector< InactiveId > parents;
      /// singleton-terminal case ({z} for a terminal world z)
      bool terminal = false;
      /// leaf index iff terminal (indexes into 'terminal_weights()'/'terminal_payoffs()')
      size_t leaf_index = npos;
   };

   /// an inactive DAG node: a set of team worlds decomposed into public observations
   struct InactiveNode {
      std::vector< NodeId > worlds;  //< ascending (memo key)
      /// the (unique) belief this node was spawned from (inactives have exactly one parent)
      BeliefId parent_belief = 0;
      /// connected components of the connectivity graph over 'worlds' -> child beliefs
      std::vector< BeliefId > components;
   };

   //////////////////////////////////////
   /// Constructors                  ///
   //////////////////////////////////////

   TeamBeliefDAG(Env env, Config config)
      requires concepts::has::method::initial_world_state< Env >
       : m_env(std::move(env)),
         m_root_state(std::make_unique< world_state_type >(m_env.initial_world_state())),
         m_config(std::move(config))
   {
      if(m_config.members.empty()) {
         throw std::invalid_argument("TeamBeliefDAG: 'members' must be non-empty");
      }
      if(m_config.max_dag_nodes == 0) {
         throw std::invalid_argument("TeamBeliefDAG: max_dag_nodes must be positive");
      }
      _build();
   }

   TeamBeliefDAG(Env env, uptr< world_state_type > root_state, Config config)
       : m_env(std::move(env)), m_root_state(std::move(root_state)), m_config(std::move(config))
   {
      if(not m_root_state) {
         throw std::invalid_argument("TeamBeliefDAG: root_state must not be null");
      }
      if(m_config.members.empty()) {
         throw std::invalid_argument("TeamBeliefDAG: 'members' must be non-empty");
      }
      if(m_config.max_dag_nodes == 0) {
         throw std::invalid_argument("TeamBeliefDAG: max_dag_nodes must be positive");
      }
      _build();
   }

   TeamBeliefDAG(const TeamBeliefDAG&) = delete;
   TeamBeliefDAG& operator=(const TeamBeliefDAG&) = delete;
   TeamBeliefDAG(TeamBeliefDAG&&) = default;
   TeamBeliefDAG& operator=(TeamBeliefDAG&&) = default;
   ~TeamBeliefDAG() = default;

   ///////////////////////////////////////////
   /// API: accessors                      ///
   ///////////////////////////////////////////

   [[nodiscard]] const Env& env() const { return m_env; }
   [[nodiscard]] const world_state_type& root_state() const { return *m_root_state; }
   [[nodiscard]] const Config& config() const { return m_config; }
   [[nodiscard]] const auto& members() const { return m_config.members; }

   /// the root participant roster (chance excluded), in stable order; indexes 'payoffs'
   [[nodiscard]] const std::vector< Player >& roster() const { return m_roster; }
   [[nodiscard]] size_t roster_index(Player player) const { return m_roster_index.at(player); }
   /// member-slot of 'player' (index into 'members()') or npos when not a team member
   [[nodiscard]] size_t slot_of(Player player) const
   {
      for(auto [slot, member] : std::views::enumerate(m_config.members)) {
         if(member == player) {
            return static_cast< size_t >(slot);
         }
      }
      return npos;
   }

   [[nodiscard]] const WorldNode& world(NodeId id) const { return m_tree.at(id); }
   [[nodiscard]] const Belief& belief(BeliefId id) const { return m_beliefs.at(id); }
   [[nodiscard]] const InactiveNode& inactive(InactiveId id) const { return m_inactives.at(id); }

   [[nodiscard]] size_t tree_node_count() const { return m_tree.size(); }
   [[nodiscard]] size_t belief_count() const { return m_beliefs.size(); }
   [[nodiscard]] size_t inactive_count() const { return m_inactives.size(); }
   [[nodiscard]] size_t node_count() const { return m_beliefs.size() + m_inactives.size(); }

   /// DAG edge count: belief->inactive prescription edges plus inactive->belief component edges
   [[nodiscard]] size_t edge_count() const
   {
      size_t out = 0;
      for(const auto& b : m_beliefs) {
         out += b.prescriptions.size();
      }
      for(const auto& o : m_inactives) {
         out += o.components.size();
      }
      return out;
   }
   /// number of prescription edges alone (the learner's regret table footprint)
   [[nodiscard]] size_t prescription_edge_count() const
   {
      size_t out = 0;
      for(const auto& b : m_beliefs) {
         out += b.prescriptions.size();
      }
      return out;
   }
   /// number of observed-decomposition edges alone
   [[nodiscard]] size_t observation_edge_count() const
   {
      size_t out = 0;
      for(const auto& o : m_inactives) {
         out += o.components.size();
      }
      return out;
   }
   [[nodiscard]] BeliefId root_belief() const { return m_root_belief; }
   [[nodiscard]] size_t terminal_count() const { return m_terminal_weights.size(); }

   /// chance probability weight p_c(z) of terminal 'leaf' (pre-folded during construction)
   [[nodiscard]] double terminal_weight(size_t leaf) const { return m_terminal_weights.at(leaf); }
   /// all leaf chance weights
   [[nodiscard]] const std::vector< double >& terminal_weights() const
   {
      return m_terminal_weights;
   }
   /// belief id of the singleton-terminal belief backing leaf 'leaf' (aligned indexing)
   [[nodiscard]] BeliefId terminal_belief_id(size_t leaf) const { return m_leaf_beliefs.at(leaf); }
   /// all leaf -> terminal-belief references
   [[nodiscard]] const std::vector< BeliefId >& terminal_belief_ids() const
   {
      return m_leaf_beliefs;
   }
   /// payoff of roster seat 'idx' at terminal 'leaf'
   [[nodiscard]] double terminal_payoff(size_t leaf, size_t roster_idx) const
   {
      return m_terminal_payoffs.at(leaf * m_roster.size() + roster_idx);
   }
   /// the world node backing terminal 'leaf' (singleton belief content)
   [[nodiscard]] NodeId terminal_world(size_t leaf) const { return m_terminal_worlds.at(leaf); }

   /// number of infosets registered for team member slot 'member_idx'
   [[nodiscard]] size_t member_infoset_count(size_t member_idx) const
   {
      return m_member_infostates.at(member_idx).ids.size();
   }
   /// the canonical representative infostate of member infoset (member_idx, local)
   [[nodiscard]] const info_state_type& member_infostate(size_t member_idx, size_t local_id) const
   {
      return *m_member_infostates.at(member_idx).representatives.at(local_id);
   }
   /// the member that owns global team infoset 'infoset_global'
   [[nodiscard]] size_t member_of_infoset(size_t infoset_global) const
   {
      return m_infoset_member.at(infoset_global);
   }
   /// member-slot owning global team infoset 'infoset_global'
   [[nodiscard]] size_t slot_of_member_infoset(size_t infoset_global) const
   {
      return m_infoset_slot.at(infoset_global);
   }
   /// member-local ordinal of global team infoset 'infoset_global'
   [[nodiscard]] size_t local_of_member_infoset(size_t infoset_global) const
   {
      return m_infoset_local.at(infoset_global);
   }
   /// canonical legal actions of global team infoset 'infoset_global'
   [[nodiscard]] const std::vector< action_type >& infoset_actions(size_t infoset_global) const
   {
      return m_infoset_actions.at(infoset_global);
   }

   ///////////////////////////////////////////////
   /// API: statistics                         ///
   ///////////////////////////////////////////////

   struct Stats {
      size_t tree_nodes = 0;
      size_t tree_leaves = 0;
      size_t tree_decision_nodes = 0;
      size_t beliefs = 0;
      size_t inactives = 0;
      size_t dag_edges = 0;
      size_t prescription_edges = 0;
      size_t observation_edges = 0;
      /// largest number of team infosets ever prescribed simultaneously at one belief
      size_t max_prescription_width = 0;
      /// largest prescription bundle at a single belief
      size_t max_prescription_fanout = 0;
      /// largest number of worlds merged into one belief
      size_t max_belief_size = 0;
   };

   [[nodiscard]] const Stats& stats() const { return m_stats; }

  private:
   ////////////////////////////
   /// build-time structures //
   ////////////////////////////

   using observation_type = auto_observation_type< Env >;

   struct MemberRegistry {
      Player player = Player::unknown;
      std::vector< sptr< info_state_type > > representatives;
      std::unordered_map<
         info_state_type,
         size_t,
         common::value_hasher< info_state_type >,
         common::value_comparator< info_state_type > >
         ids;
      /// local infoset ordinal -> global team-infoset id
      std::vector< size_t > local_ids;
   };

   struct LabelHash {
      size_t operator()(const std::vector< NodeId >& label) const noexcept
      {
         size_t seed = label.size();
         for(auto id : label) {
            common::hash_combine(seed, std::hash< NodeId >{}(id));
         }
         return seed;
      }
   };
   using LabelMap = std::unordered_map< std::vector< NodeId >, size_t, LabelHash >;

   /// undo record of one edge advance of the enumeration traversal
   struct EdgeUndo {
      bool flushes = false;
      Player flush_target = Player::unknown;
      sptr< info_state_type > saved_infostate{};
      std::optional< std::vector< std::pair< observation_type, observation_type > > >
         saved_flush_buffer{};
      std::vector< std::pair< Player, size_t > > saved_sizes{};
   };

   ///////////////////////
   /// internals        //
   ///////////////////////

   Env m_env;
   uptr< world_state_type > m_root_state;
   Config m_config;

   std::vector< Player > m_roster{};
   player_hashmap< size_t > m_roster_index{};
   std::vector< MemberRegistry > m_member_infostates{};
   std::vector< Player > m_infoset_owner{};  //< global infoset id -> owning member
   std::vector< size_t > m_infoset_member{};  //< global infoset id -> member slot
   std::vector< size_t > m_infoset_slot{};
   std::vector< size_t > m_infoset_local{};
   std::vector< std::vector< action_type > > m_infoset_actions{};

   std::vector< WorldNode > m_tree{};
   std::vector< Belief > m_beliefs{};
   std::vector< InactiveNode > m_inactives{};
   LabelMap m_belief_ids{};
   LabelMap m_inactive_ids{};
   BeliefId m_root_belief = 0;
   size_t m_infoset_global_count = 0;

   std::vector< double > m_terminal_weights{};
   std::vector< double > m_terminal_payoffs{};
   std::vector< NodeId > m_terminal_worlds{};
   std::unordered_map< NodeId, size_t > m_leaf_by_world{};
   /// leaf row -> singleton terminal belief id (filled at the end of _build)
   std::vector< BeliefId > m_leaf_beliefs{};

   Stats m_stats{};

   /// one-shot arena bookkeeping of the enumeration (same discipline as the house ICFR)
   std::deque< utils::ReusableSlot< world_state_type > > m_arena{};
   player_hashmap< std::vector< std::pair< observation_type, observation_type > > > m_obs_buffers{};
   player_hashmap< sptr< info_state_type > > m_active_infostates{};

   ///////////////////////
   /// construction     //
   ///////////////////////

   void _build();
   /// creates tree node 'node_id' (preset by the caller with depth/chance fields) and expands
   /// its subtree
   void _enumerate_visit(world_state_type& state, NodeId node_id, size_t depth);
   /// bottom-up ancestor-set reduction over the finished preorder enumeration + leaf tables
   void _finalize_enumeration();

   /// advances an arena copy of 'state' along 'edge', folding observations into the tracked
   /// infostates (mirrors the house ICFR edge mechanics); restore via '_undo_edge'
   template < typename ActionOrOutcome >
   world_state_type& _advance_edge(
      const world_state_type& state,
      size_t depth,
      const ActionOrOutcome& edge,
      EdgeUndo& undo
   );
   void _undo_edge(const EdgeUndo& undo);

   /// paper Algorithm 1 workers; label vectors are canonicalized internally
   BeliefId _make_active(std::vector< NodeId > b);
   InactiveId _make_inactive(std::vector< NodeId > o, BeliefId parent_belief);
};

}  // namespace nor::rm::team

// include the actual template implementations of this class
#include "team_belief_dag.tcc"

#endif  // NOR_RM_TEAM_TEAM_BELIEF_DAG_HPP


#ifndef NOR_PURSUIT_EVASION_STATE_HPP
#define NOR_PURSUIT_EVASION_STATE_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <variant>
#include <vector>

namespace pursuit_evasion {

/// 'none' marks the absence of an acting player (terminal states); its value mirrors
/// nor::Player::unknown so the FOSG adapter's cast stays consistent.
/// Player::one is the ATTACKER, Player::two the DEFENDER.
enum class Player : int8_t { none = -2, one = 0, two = 1 };

[[nodiscard]] constexpr Player opponent(Player player)
{
   return player == Player::one ? Player::two : Player::one;
}

template < typename To = size_t, typename T >
[[nodiscard]] constexpr To as_int(T value)
{
   return static_cast< To >(value);
}

// #####################################################################################################################
// graph transcription of PCFR+ App. G, Fig. 3 (arXiv:2007.14358)
// #####################################################################################################################
//
// The figure's tikz source places 13 circle nodes:
//   A=S at x=0 (leftmost start), columns B,C,D at x=1.2 / E,F,G at x=2.4 / H,I,J at x=3.6 with
//   y = +1,0,-1 top-to-bottom, and exits K,L,M at x=4.8 labelled 5,10,3 top-to-bottom.
// NOTE: the task brief calls this a "7-node graph"; the verbatim figure has 13 nodes and the
// verbatim-faithful transcription below follows the paper. Black directed edges are attacker-only;
// grey dashed edges are patrol-only. Two drawing facts are explicit in the tikz source and worth
// citing because they are easy to misread from a rendered figure:
//   - the middle column merges one-way into its centre: E->F and G->F exist but F->E/F->G do not,
//     and there are no lateral edges among B/C/D or H/I/J for the attacker (those pairs are linked
//     by the grey dashed patrol lines only);
//   - the grey dashed lines are B-C, C-D (patrol area P1) and H-I, I-J (patrol area P2); the shaded
//     rectangles cover exactly those two columns.

/// node count of the transcribed graph (S + 3x3 interior + 3 exits)
constexpr size_t node_count = 13;

/// node ids following the tikz labels of Fig. 3
inline constexpr uint8_t node_S = 0;  //< "S", the leftmost start node
inline constexpr uint8_t node_B = 1;  //< column 1 top    (inside patrol area P1)
inline constexpr uint8_t node_C = 2;  //< column 1 centre (inside patrol area P1)
inline constexpr uint8_t node_D = 3;  //< column 1 bottom (inside patrol area P1)
inline constexpr uint8_t node_E = 4;  //< column 2 top
inline constexpr uint8_t node_F = 5;  //< column 2 centre
inline constexpr uint8_t node_G = 6;  //< column 2 bottom
inline constexpr uint8_t node_H = 7;  //< column 3 top    (inside patrol area P2)
inline constexpr uint8_t node_I = 8;  //< column 3 centre (inside patrol area P2)
inline constexpr uint8_t node_J = 9;  //< column 3 bottom (inside patrol area P2)
inline constexpr uint8_t node_K = 10;  //< exit with escape payoff 5 (top right)
inline constexpr uint8_t node_L = 11;  //< exit with escape payoff 10 (middle right)
inline constexpr uint8_t node_M = 12;  //< exit with escape payoff 3 (bottom right)

/// a directed graph edge
struct Edge {
   uint8_t from = 0;
   uint8_t to = 0;

   friend bool operator==(const Edge&, const Edge&) = default;
   friend bool operator!=(const Edge&, const Edge&) = default;
};

/**
 * @brief The black directed attacker edges of Fig. 3 in tikz order.
 *
 * Transcription note: the figure draws exactly these 14 arrows. In particular S has no incoming
 * edges, the exits have no outgoing edges (reaching one ends the game) and the E->F / G->F merge
 * is one-way.
 */
inline constexpr std::array< Edge, 14 > k_attacker_edges{
   {Edge{node_S, node_B},
    Edge{node_S, node_C},
    Edge{node_S, node_D},
    Edge{node_B, node_E},
    Edge{node_C, node_F},
    Edge{node_D, node_G},
    Edge{node_E, node_F},
    Edge{node_G, node_F},
    Edge{node_E, node_H},
    Edge{node_F, node_I},
    Edge{node_G, node_J},
    Edge{node_H, node_K},
    Edge{node_I, node_L},
    Edge{node_J, node_M}}};

namespace _detail {

[[nodiscard]] inline constexpr std::array< uint16_t, node_count > _out_masks()
{
   std::array< uint16_t, node_count > masks{};
   for(const auto& edge : k_attacker_edges) {
      masks[edge.from] |= uint16_t(1) << edge.to;
   }
   return masks;
}

}  // namespace _detail

/// bit t of k_attacker_out[n] <=> a black directed edge n -> t exists
inline constexpr std::array< uint16_t, node_count > k_attacker_out = _detail::_out_masks();

[[nodiscard]] constexpr bool has_attacker_edge(uint8_t from, uint8_t to)
{
   return ((k_attacker_out[from] >> to) & uint16_t(1)) != 0;
}

/// the nodes of patrol area P1 (the figure's left shaded column)
inline constexpr std::array< uint8_t, 3 > k_patrol1_area{{node_B, node_C, node_D}};

/// the nodes of patrol area P2 (the figure's right shaded column)
inline constexpr std::array< uint8_t, 3 > k_patrol2_area{{node_H, node_I, node_J}};

/**
 * @brief The grey dashed patrol edges of Fig. 3.
 *
 * Undirected by construction; each connects two nodes of ONE patrol area, so patrols can never
 * leave their respective areas. The middle column and the exits belong to no patrol area.
 */
inline constexpr std::array< Edge, 4 > k_patrol_edges{
   {Edge{node_B, node_C}, Edge{node_C, node_D}, Edge{node_H, node_I}, Edge{node_I, node_J}}};

[[nodiscard]] inline bool has_patrol_edge(uint8_t a, uint8_t b)
{
   for(const auto& edge : k_patrol_edges) {
      if((edge.from == a && edge.to == b) || (edge.from == b && edge.to == a)) {
         return true;
      }
   }
   return false;
}

[[nodiscard]] constexpr bool in_patrol1(uint8_t node)
{
   return node == node_B || node == node_C || node == node_D;
}

[[nodiscard]] constexpr bool in_patrol2(uint8_t node)
{
   return node == node_H || node == node_I || node == node_J;
}

/// the rightmost exit nodes, top-to-bottom as drawn in Fig. 3
inline constexpr std::array< uint8_t, 3 > k_exit_nodes{{node_K, node_L, node_M}};

/// attacker escape payoffs of the exits, top-to-bottom (5 above, 10 centre, 3 below)
inline constexpr std::array< double, 3 > k_exit_payoffs{{5., 10., 3.}};

[[nodiscard]] constexpr bool is_exit(uint8_t node)
{
   return node == node_K || node == node_L || node == node_M;
}

/// index into k_exit_nodes / k_exit_payoffs of an exit node
[[nodiscard]] constexpr size_t exit_index(uint8_t node)
{
   switch(node) {
      case node_K: return 0;
      case node_L: return 1;
      case node_M: return 2;
      default: throw std::invalid_argument("pursuit evasion: node is not an exit.");
   }
}

/// initial positions. TRANSCRIPTION ASSUMPTION: the figure marks no starting squares for the
/// patrols; each patrol starts on the centre node of its area (C resp. I), which is the unique
/// symmetric choice. The attacker starts on S as stated in the paper text.
inline constexpr uint8_t k_initial_attacker_node = node_S;
inline constexpr std::array< uint8_t, 2 > k_initial_patrol_nodes{{node_C, node_I}};

[[nodiscard]] constexpr uint16_t trace_bit(uint8_t node)
{
   return uint16_t(1) << node;
}

// #####################################################################################################################
// actions & configuration
// #####################################################################################################################

/**
 * @brief One simultaneous-move commitment of the attacker.
 *
 * 'edge_id' indexes k_attacker_edges and must originate at the attacker's current node; the
 * sentinel wait_edge_id encodes waiting in place, which additionally clears ALL his traces.
 */
struct AttMove {
   static constexpr uint8_t wait_edge_id = 255;

   uint8_t edge_id = wait_edge_id;

   [[nodiscard]] bool is_wait() const { return edge_id == wait_edge_id; }

   friend bool operator==(const AttMove&, const AttMove&) = default;
   friend bool operator!=(const AttMove&, const AttMove&) = default;
};

/**
 * @brief One compound simultaneous-move commitment of the defender.
 *
 * The defender moves BOTH patrols in one turn ('p1' and 'p2' target nodes); each target must be
 * the patrol's current node or reachable along a grey dashed edge.
 */
struct DefMove {
   uint8_t p1 = 0;  //< patrol-1 target node (in k_patrol1_area)
   uint8_t p2 = 0;  //< patrol-2 target node (in k_patrol2_area)

   friend bool operator==(const DefMove&, const DefMove&) = default;
   friend bool operator!=(const DefMove&, const DefMove&) = default;
};

using Action = std::variant< AttMove, DefMove >;

/**
 * @brief The phase of a round.
 *
 * The paper's simultaneous move is sequentialized commit-commit-resolve (repo convention, cf.
 * goofspiel): CommitA -> CommitD -> Resolve. Because this environment is DETERMINISTIC
 * (Stochasticity::deterministic, monostate chance outcome), the resolve carries a single degenerate
 * outcome and is fused into the application of the defender's commitment: applying DefMove applies
 * both commitments simultaneously at once. This yields the identical game tree minus a trivial
 * one-outcome node, while keeping every non-terminal state attributable to a real acting player
 * (which deterministic-env CFR traversal requires).
 */
enum class Phase : uint8_t { commit_attacker = 0, commit_defender = 1 };

/// how a game instance ended
enum class TerminalCause : uint8_t { none = 0, capture, escape, timeout };

/**
 * @brief Configuration of a pursuit-evasion instance.
 *
 * @param rounds m, the number of simultaneous rounds before timeout; the paper benchmarks m=4, 5
 * and 6 (headline curves use m=6). Reaching m resolved rounds without capture or escape ends the
 * game in a 0/0 timeout.
 */
struct Config {
   /// hard cap on 'rounds' so the fixed-size round log stays bounded
   static constexpr size_t max_rounds = 8;

   size_t rounds = 6;

   Config() = default;
   explicit Config(size_t rounds_) : rounds(rounds_) { validate(); }

   void validate() const
   {
      if(rounds < 1 or rounds > max_rounds) {
         throw std::invalid_argument(
            "pursuit evasion supports 1 to " + std::to_string(max_rounds) + " rounds (got "
            + std::to_string(rounds) + ")."
         );
      }
   }

   [[nodiscard]] friend bool operator==(const Config&, const Config&) = default;
};

/**
 * @brief One chronological entry of the world state's round log.
 *
 * Needed to reconstruct observation streams (histories): a resolved round's trace sightings are
 * derived information not recoverable from the bare moves.
 */
struct RoundRecord {
   int16_t att_edge = -1;  //< attacker edge id taken this round; -1 <=> waited in place
   int8_t p1_to = 0;  //< patrol-1 target node
   int8_t p2_to = 0;  //< patrol-2 target node
   bool sight1 = false;  //< patrol-1 ENTERED (moved onto) a node carrying a pre-round trace
   bool sight2 = false;  //< patrol-2 ... analogously
   TerminalCause cause = TerminalCause::none;  //< non-none only in the final record

   friend bool operator==(const RoundRecord&, const RoundRecord&) = default;
   friend bool operator!=(const RoundRecord&, const RoundRecord&) = default;
};

/**
 * @brief The world state of one deterministic pursuit-evasion game.
 *
 * All data lives in fixed-size members so copying (which happens for every edge of the search
 * tree) stays cheap. Trace semantics (paper App. G): every position the attacker arrives on leaves
 * a trace unless he waited that round, in which case ALL his traces are cleared; the initial spawn
 * on S deposits no trace since no action was taken there. Sightings never capture -- only
 * co-location does.
 */
class State {
  public:
   explicit State(Config config = {}) : m_config(config)
   {
      m_config.validate();
      m_attacker_node = k_initial_attacker_node;
      m_patrol_nodes = k_initial_patrol_nodes;
   }

   ////////////////////////////////
   /// API: transitions        ///
   ////////////////////////////////

   void apply_action(const Action& action)
   {
      if(const auto* att = std::get_if< AttMove >(&action)) {
         apply_action(*att);
         return;
      }
      apply_action(std::get< DefMove >(action));
   }

   /**
    * Commits the attacker's move (CommitA phase). Positions stay untouched until the defender
    * commits; the joint resolution happens atomically inside apply_action(DefMove).
    */
   void apply_action(AttMove move)
   {
      if(terminal()) {
         throw std::logic_error("pursuit evasion state is terminal; no further actions.");
      }
      if(m_phase != Phase::commit_attacker) {
         throw std::invalid_argument(
            "pursuit evasion: attacker commitments are only legal during CommitA (got phase "
            + std::to_string(unsigned(m_phase)) + ")."
         );
      }
      if(not is_valid(move)) {
         throw std::invalid_argument(
            "pursuit evasion: attacker move illegal at current node (edge "
            + std::to_string(unsigned(move.edge_id)) + ")."
         );
      }
      m_committed_att = move;
      m_phase = Phase::commit_defender;
      m_active = Player::two;
   }

   /**
    * Commits the defender's compound move and resolves the round: both commitments are applied
    * SIMULTANEOUSLY, then capture (post-move co-location with either patrol) is checked, then
    * escape (attacker stands on an exit), then traces/sightings are settled and finally the m-round
    * timeout. Sightings are computed against the PRE-round trace mask so that simultaneity never
    * lets a patrol react to, or benefit from, the very trace deposited in the same tick.
    */
   void apply_action(DefMove move)
   {
      if(terminal()) {
         throw std::logic_error("pursuit evasion state is terminal; no further actions.");
      }
      if(m_phase != Phase::commit_defender) {
         throw std::invalid_argument(
            "pursuit evasion: defender commitments are only legal during CommitD (got phase "
            + std::to_string(unsigned(m_phase)) + ")."
         );
      }
      if(not is_valid(move)) {
         throw std::invalid_argument("pursuit evasion: defender compound move illegal.");
      }
      const auto committed_att = *m_committed_att;
      const bool att_waited = committed_att.is_wait();
      m_committed_def = move;

      // --- simultaneous application of both commitments ---
      const uint16_t pre_traces = m_trace_mask;
      const auto [p1_from, p2_from] = m_patrol_nodes;
      if(not att_waited) {
         m_attacker_node = k_attacker_edges[committed_att.edge_id].to;
      }
      m_patrol_nodes = {move.p1, move.p2};
      // a sighting fires only when a patrol ENTERS (not stays on) a node carrying a pre-round
      // trace -- paper wording: "if a patrol VISITS a node that was previously visited by the
      // attacker". (Staying on an already-traced node cannot hide fresh information: the first
      // entry fired the signal.)
      const bool sight1 = move.p1 != p1_from && ((pre_traces >> move.p1) & uint16_t(1)) != 0;
      const bool sight2 = move.p2 != p2_from && ((pre_traces >> move.p2) & uint16_t(1)) != 0;

      auto& record = m_rounds[m_round];
      record.att_edge = att_waited ? int16_t(-1) : int16_t(committed_att.edge_id);
      record.p1_to = int8_t(move.p1);
      record.p2_to = int8_t(move.p2);
      record.sight1 = sight1;
      record.sight2 = sight2;

      auto captured = [&] {
         return m_attacker_node == m_patrol_nodes[0] || m_attacker_node == m_patrol_nodes[1];
      }();

      if(captured) {
         record.cause = m_terminal_cause = TerminalCause::capture;
      } else if(is_exit(m_attacker_node)) {
         record.cause = m_terminal_cause = TerminalCause::escape;
      } else {
         // trace bookkeeping: waiting cleans every trace, moving deposits one at the arrival node
         if(att_waited) {
            m_trace_mask = 0;
         } else {
            m_trace_mask |= trace_bit(m_attacker_node);
         }
         m_round += 1;
         if(m_round >= m_config.rounds) {
            record.cause = m_terminal_cause = TerminalCause::timeout;
         } else {
            record.cause = TerminalCause::none;
            m_committed_att.reset();
            m_committed_def.reset();
            m_phase = Phase::commit_attacker;
            m_active = Player::one;
            return;
         }
      }
      m_active = Player::none;
   }

   [[nodiscard]] bool is_valid(AttMove move) const
   {
      if(move.is_wait()) {
         return true;
      }
      return move.edge_id < k_attacker_edges.size()
             && k_attacker_edges[move.edge_id].from == m_attacker_node;
   }

   [[nodiscard]] bool is_valid(DefMove move) const
   {
      auto legal_step = [&](uint8_t from, uint8_t to, auto in_area) {
         if(not in_area(to)) {
            return false;
         }
         return to == from || has_patrol_edge(from, to);
      };
      return legal_step(m_patrol_nodes[0], move.p1, [](uint8_t n) { return in_patrol1(n); })
             && legal_step(m_patrol_nodes[1], move.p2, [](uint8_t n) { return in_patrol2(n); });
   }

   [[nodiscard]] bool is_valid(const Action& action) const
   {
      if(terminal()) {
         return false;
      }
      if(const auto* att = std::get_if< AttMove >(&action)) {
         return m_phase == Phase::commit_attacker && is_valid(*att);
      }
      const auto* def = std::get_if< DefMove >(&action);
      return def != nullptr && m_phase == Phase::commit_defender && is_valid(*def);
   }

   ////////////////////////////////
   /// API: queries            ///
   ////////////////////////////////

   [[nodiscard]] bool terminal() const { return m_terminal_cause != TerminalCause::none; }

   [[nodiscard]] TerminalCause terminal_cause() const { return m_terminal_cause; }

   [[nodiscard]] bool timed_out() const { return m_terminal_cause == TerminalCause::timeout; }

   /// zero-sum terminal payoffs: capture -1/+1, escape +v/-v (v in {5,10,3}), timeout 0/0
   [[nodiscard]] double payoff(Player player) const
   {
      double attacker_value = 0.;
      switch(m_terminal_cause) {
         case TerminalCause::capture: attacker_value = -1.; break;
         case TerminalCause::escape:
            attacker_value = k_exit_payoffs[exit_index(m_attacker_node)];
            break;
         case TerminalCause::timeout:
         case TerminalCause::none: attacker_value = 0.; break;
      }
      return player == Player::one ? attacker_value : -attacker_value;
   }

   [[nodiscard]] std::array< double, 2 > payoffs() const
   {
      return {payoff(Player::one), payoff(Player::two)};
   }

   /// legal actions of `player`; empty outside their commitment phase
   [[nodiscard]] std::vector< Action > actions(Player player) const
   {
      std::vector< Action > out;
      if(terminal() or m_active != player) {
         return out;
      }
      if(m_phase == Phase::commit_attacker && player == Player::one) {
         out.reserve(k_attacker_edges.size());
         for(size_t e = 0; e < k_attacker_edges.size(); ++e) {
            if(k_attacker_edges[e].from == m_attacker_node) {
               out.emplace_back(Action{AttMove{uint8_t(e)}});
            }
         }
         out.emplace_back(Action{AttMove{AttMove::wait_edge_id}});
      } else if(m_phase == Phase::commit_defender && player == Player::two) {
         out.reserve(k_patrol1_area.size() * k_patrol2_area.size());
         for(uint8_t t1 : k_patrol1_area) {
            if(not (t1 == m_patrol_nodes[0] || has_patrol_edge(m_patrol_nodes[0], t1))) {
               continue;
            }
            for(uint8_t t2 : k_patrol2_area) {
               if(t2 == m_patrol_nodes[1] || has_patrol_edge(m_patrol_nodes[1], t2)) {
                  out.emplace_back(Action{DefMove{t1, t2}});
               }
            }
         }
      }
      return out;
   }

   [[nodiscard]] Player active_player() const { return m_active; }
   [[nodiscard]] const Config& config() const { return m_config; }
   [[nodiscard]] Phase phase() const { return m_phase; }
   /// number of fully resolved rounds
   [[nodiscard]] size_t round() const { return m_round; }
   [[nodiscard]] uint8_t attacker_node() const { return m_attacker_node; }
   [[nodiscard]] const std::array< uint8_t, 2 >& patrol_nodes() const { return m_patrol_nodes; }
   /// bit n <=> the attacker visited-and-not-cleaned node n (see class comment)
   [[nodiscard]] uint16_t trace_mask() const { return m_trace_mask; }
   [[nodiscard]] std::optional< AttMove > committed_attacker_move() const
   {
      return m_committed_att;
   }
   [[nodiscard]] std::optional< DefMove > committed_defender_move() const
   {
      return m_committed_def;
   }
   /// the per-round records; entries at indices >= round() are not finalized yet
   [[nodiscard]] const std::array< RoundRecord, Config::max_rounds >& rounds() const
   {
      return m_rounds;
   }

   [[nodiscard]] bool operator==(const State& other) const
   {
      return m_config == other.m_config && m_active == other.m_active && m_phase == other.m_phase
             && m_attacker_node == other.m_attacker_node && m_patrol_nodes == other.m_patrol_nodes
             && m_trace_mask == other.m_trace_mask && m_round == other.m_round
             && m_terminal_cause == other.m_terminal_cause
             && m_committed_att == other.m_committed_att && m_committed_def == other.m_committed_def
             && m_rounds == other.m_rounds;
   }
   [[nodiscard]] bool operator!=(const State& other) const { return not (*this == other); }

  private:
   Config m_config{};
   Player m_active = Player::one;
   Phase m_phase = Phase::commit_attacker;
   uint8_t m_attacker_node = k_initial_attacker_node;
   std::array< uint8_t, 2 > m_patrol_nodes = k_initial_patrol_nodes;
   uint16_t m_trace_mask = 0;
   size_t m_round = 0;
   TerminalCause m_terminal_cause = TerminalCause::none;
   std::optional< AttMove > m_committed_att{};
   std::optional< DefMove > m_committed_def{};
   std::array< RoundRecord, Config::max_rounds > m_rounds{};
};

}  // namespace pursuit_evasion

#endif  // NOR_PURSUIT_EVASION_STATE_HPP

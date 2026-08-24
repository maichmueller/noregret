
#ifndef NOR_COLONEL_BLOTTO_STATE_HPP
#define NOR_COLONEL_BLOTTO_STATE_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

namespace colonel_blotto {

/// 'none' marks the absence of an acting player (terminal states); its value mirrors
/// nor::Player::unknown so the FOSG adapter's cast stays consistent (cf. pursuit_evasion /
/// oshi_zumo).
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
// rule transcription notes (canonical literature source)
// #####################################################################################################################
//
// SOURCE. Colonel Blotto games originate with Gross & Wagner (1950), "A Continuous Colonel
// Blotto Game" (RAND/P-258; the folklore dates the puzzle to Gross & Wagner's 1953 RAND note and
// even earlier military parlor games attributed to Col. Blotto in the 1920s press). The
// DISCRETIZED finite version used here is the standard CFR testbed formulation (cf. OpenSpiel
// open_spiel/games/blotto/blotto.cc -- simultaneous one-shot, deterministic, zero-sum, default
// fields=3; and Arjona-Medina et al./Behnezhad et al. for the discrete problem statement):
// each of two players distributes an integer budget across N battlefields of equal value
// v_j = 1; on each battlefield the STRICTLY larger force wins that battlefield's value, equal
// forces SPLIT it (each side receives v_j / 2); a player's terminal payoff is the FRACTION of the
// total value he won, i.e. (#battlefields won + 0.5 * #ties) / N.
//
// ZERO-SUM AFTER CENTERING. Fractions always sum to 1, hence reward(player) := frac_won(player)
// - 1/N sums to exactly 0 across players at every terminal history: the game is CONSTANT-SUM,
// and the normalized metric exploitability(..., constant_sum=true) is well-defined (unlike the
// general-sum Shapley/centipede entries of this suite).
//
// SEQUENTIALIZATION (strategic-equivalence caveat). The simultaneous vector allocation is
// sequentialized FIELD-BY-FIELD as CommitP1(f1) -> CommitP2(f1) -> CommitP1(f2) -> ... ->
// CommitP2(fN), with the deterministic all-field resolution fused into the final CommitP2
// application (oshi_zumo convention; keeps every non-terminal state attributable to a real acting
// player). Commitments are SECRET: no allocation value is observable before the final reveal, so
// neither player ever conditions on new information between his own commitments. By Kuhn's
// theorem the behavioral-strategy game therefore realizes exactly the same normal-form strategy
// sets (any distribution over full vectors is implementable by coordinate-wise mixing) and the
// same expected-payoff pairs as the simultaneous game, because each battlefield's outcome depends
// only on the MARGINAL distributions of the two allocations on that field. Caveat worth keeping
// in mind when comparing infoset counts against flat simultaneous encodings: the sequentialized
// tree splits one joint decision into 2*N commitment stages with growing action spaces
// (remaining_budget + 1 legal deployments per stage).

constexpr size_t battlefield_count = 3;

/// hard cap keeping the uint32 bookkeeping and tree size comfortable
constexpr size_t max_budget = 8;

/**
 * @brief Configuration of a discretized Blotto instance.
 *
 * @param budget integer troops B each player distributes across the battlefields; a deployment
 *        of x troops on a field consumes x from the player's remaining budget (sum <= B, unspent
 *        budget is discarded -- the standard generous discretization)
 */
struct BlottoConfig {
   size_t budget = 3;

   BlottoConfig() = default;
   explicit BlottoConfig(size_t budget_) : budget(budget_) { validate(); }

   void validate() const
   {
      if(budget < 1 or budget > max_budget) {
         throw std::invalid_argument(
            "colonel blotto supports budgets of 1 to " + std::to_string(max_budget) + " (got "
            + std::to_string(budget) + ")."
         );
      }
   }

   [[nodiscard]] friend bool operator==(const BlottoConfig&, const BlottoConfig&) = default;
};

/// one commitment: the number of troops deployed on the current battlefield
struct Deploy {
   uint32_t troops = 0;

   friend bool operator==(const Deploy&, const Deploy&) = default;
   friend bool operator!=(const Deploy&, const Deploy&) = default;
};

/// outcome of one resolved battlefield (strict winner takes the full unit value, tie splits it)
enum class FieldOutcome : uint8_t { one_wins = 0, split = 1, two_wins = 2 };

/// how a game instance ended
enum class TerminalCause : uint8_t { none = 0, resolved };

/**
 * @brief The phase of the sequentialized joint allocation.
 *
 * CommitP1 -> CommitP2 per battlefield with the deterministic all-field resolution fused into the
 * FINAL CommitP2 application (repo convention for deterministic environments, cf. oshi_zumo).
 */
enum class Phase : uint8_t { commit_p1 = 0, commit_p2 = 1 };

/**
 * @brief World state of one discretized Colonel Blotto game.
 *
 * All data lives in fixed-size members so copying (which happens for every edge of the search
 * tree) stays cheap.
 */
class State {
  public:
   explicit State(BlottoConfig config = {}) : m_config(config)
   {
      m_config.validate();
      m_remaining = {uint32_t(m_config.budget), uint32_t(m_config.budget)};
   }

   ////////////////////////////////
   /// API: transitions        ///
   ////////////////////////////////

   /**
    * Commits a deployment for whoever is due on the current battlefield. During CommitP1 nothing
    * but the commitment is stored; when applied to the LAST battlefield of player two the joint
    * resolution happens atomically (fused deterministic resolve, see Phase).
    */
   void apply_action(Deploy deploy)
   {
      if(terminal()) {
         throw std::logic_error("colonel blotto: state is terminal; no further deployments.");
      }
      switch(m_phase) {
         case Phase::commit_p1: {
            if(not is_valid(Player::one, deploy)) {
               throw std::invalid_argument(
                  "colonel blotto: illegal player-one deployment " + std::to_string(deploy.troops)
                  + " (remaining " + std::to_string(m_remaining.at(as_int(Player::one))) + ")."
               );
            }
            _commit(Player::one, deploy);
            m_phase = Phase::commit_p2;
            m_active = Player::two;
            return;
         }
         case Phase::commit_p2: {
            if(not is_valid(Player::two, deploy)) {
               throw std::invalid_argument(
                  "colonel blotto: illegal player-two deployment " + std::to_string(deploy.troops)
                  + " (remaining " + std::to_string(m_remaining.at(as_int(Player::two))) + ")."
               );
            }
            _commit(Player::two, deploy);
            if(m_field + 1 < battlefield_count) {
               m_field += 1;
               m_phase = Phase::commit_p1;
               m_active = Player::one;
            } else {
               _resolve_all();
            }
            return;
         }
      }
      throw std::logic_error("colonel blotto: unreachable phase.");
   }

   [[nodiscard]] bool is_valid(Deploy deploy) const
   {
      if(terminal()) {
         return false;
      }
      switch(m_phase) {
         case Phase::commit_p1: return is_valid(Player::one, deploy);
         case Phase::commit_p2: return is_valid(Player::two, deploy);
      }
      return false;
   }

   /// legality of `deploy` for `player` independent of the current phase (cannot overspend)
   [[nodiscard]] bool is_valid(Player player, Deploy deploy) const
   {
      switch(player) {
         case Player::one:
         case Player::two: return deploy.troops <= m_remaining.at(as_int(player));
         default: return false;
      }
   }

   /// legal deployments of whoever is due to commit (empty outside the commit phases / terminal)
   [[nodiscard]] std::vector< Deploy > actions(Player player) const
   {
      std::vector< Deploy > out;
      if(terminal() or m_active != player) {
         return out;
      }
      out.reserve(m_remaining.at(as_int(player)) + 1);
      for(uint32_t t = 0; t <= m_remaining.at(as_int(player)); ++t) {
         out.push_back(Deploy{t});
      }
      return out;
   }

   ////////////////////////////////
   /// API: queries            ///
   ////////////////////////////////

   [[nodiscard]] bool terminal() const { return m_terminal_cause != TerminalCause::none; }
   [[nodiscard]] TerminalCause terminal_cause() const { return m_terminal_cause; }

   /**

    * @brief CONSTANT-SUM (zero-sum after centering) terminal payoffs.
    *
    * reward(player) = fraction_of_total_value_won(player) - 1/2, with fractions computed as
    * (#battlefields won + 0.5 * #splits) / battlefield_count under uniform v_j = 1. Rewards sum
    * to exactly 0 across players at every terminal history. 0 before terminality.
    */
   [[nodiscard]] double payoff(Player player) const
   {
      if(not terminal()) {
         return 0.;
      }
      const double frac_won = double(m_won.at(as_int(player)));
      return frac_won / double(battlefield_count) - 0.5;
   }

   [[nodiscard]] std::array< double, 2 > payoffs() const
   {
      return {payoff(Player::one), payoff(Player::two)};
   }

   [[nodiscard]] Player active_player() const { return m_active; }
   [[nodiscard]] const BlottoConfig& config() const { return m_config; }
   [[nodiscard]] Phase phase() const { return m_phase; }
   /// index of the battlefield currently being committed ([0, battlefield_count); stale at
   /// terminality where it stays at the last field)
   [[nodiscard]] size_t field() const { return m_field; }
   [[nodiscard]] uint32_t remaining_budget(Player player) const
   {
      return m_remaining.at(as_int(player));
   }
   /// nullopt while the respective side has not committed this battlefield yet
   [[nodiscard]] std::optional< Deploy > committed_deploy(Player player) const
   {
      return m_committed.at(as_int(player));
   }
   /// the full allocation vectors (valid only once terminal)
   [[nodiscard]] std::
      pair< std::array< uint32_t, battlefield_count >, std::array< uint32_t, battlefield_count > >
      allocations() const
   {
      return {m_allocations.at(as_int(Player::one)), m_allocations.at(as_int(Player::two))};
   }
   /// per-battlefield outcomes in field order (valid only once terminal)
   [[nodiscard]] const std::array< FieldOutcome, battlefield_count >& field_outcomes() const
   {
      return m_outcomes;
   }
   /// raw won-value counters (won + half ties), pre-normalization
   [[nodiscard]] double won_value(Player player) const { return m_won.at(as_int(player)); }

   [[nodiscard]] bool operator==(const State& other) const
   {
      return m_config == other.m_config && m_remaining == other.m_remaining
             && m_allocations == other.m_allocations && m_committed == other.m_committed
             && m_field == other.m_field && m_phase == other.m_phase && m_active == other.m_active
             && m_terminal_cause == other.m_terminal_cause && m_outcomes == other.m_outcomes
             && m_won == other.m_won;
   }
   [[nodiscard]] bool operator!=(const State& other) const { return not (*this == other); }

  private:
   /////////////////////////////////
   /// API: private utilities   ///
   /////////////////////////////////

   void _commit(Player mover, Deploy deploy)
   {
      m_committed.at(as_int(mover)) = deploy;
      m_allocations.at(as_int(mover)).at(m_field) = deploy.troops;
      m_remaining.at(as_int(mover)) -= deploy.troops;
   }

   /// fused deterministic resolution of ALL battlefields (uniform v_j = 1)
   void _resolve_all()
   {
      double won_one = 0.;
      double won_two = 0.;
      const auto& alloc_one = m_allocations.at(as_int(Player::one));
      const auto& alloc_two = m_allocations.at(as_int(Player::two));
      for(size_t j = 0; j < battlefield_count; ++j) {
         if(alloc_one.at(j) > alloc_two.at(j)) {
            m_outcomes.at(j) = FieldOutcome::one_wins;
            won_one += 1.;
         } else if(alloc_two.at(j) > alloc_one.at(j)) {
            m_outcomes.at(j) = FieldOutcome::two_wins;
            won_two += 1.;
         } else {
            m_outcomes.at(j) = FieldOutcome::split;
            won_one += 0.5;
            won_two += 0.5;
         }
      }
      m_won = {won_one, won_two};
      m_terminal_cause = TerminalCause::resolved;
      m_active = Player::none;
   }

   /////////////////////////////////
   /// API: data members        ///
   /////////////////////////////////

   BlottoConfig m_config{};
   std::array< uint32_t, 2 > m_remaining{};
   std::array< std::array< uint32_t, battlefield_count >, 2 > m_allocations{};
   std::array< std::optional< Deploy >, 2 > m_committed{};
   size_t m_field = 0;
   Phase m_phase = Phase::commit_p1;
   Player m_active = Player::one;
   TerminalCause m_terminal_cause = TerminalCause::none;
   std::array< FieldOutcome, battlefield_count > m_outcomes{};
   std::array< double, 2 > m_won{};
};

}  // namespace colonel_blotto

#endif  // NOR_COLONEL_BLOTTO_STATE_HPP

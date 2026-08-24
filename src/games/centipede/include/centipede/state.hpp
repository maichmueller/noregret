
#ifndef NOR_CENTIPEDE_STATE_HPP
#define NOR_CENTIPEDE_STATE_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

namespace centipede {

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
// SOURCE. Rosenthal, R. W. (1981), "Games of Perfect Information, Predatory Pricing and the Chain
// Store Paradox", Journal of Economic Theory 25(1), 92-100 -- the centipede game; experimental
// benchmark: McKelvey & Palfrey (1992), "An Experimental Study of the Centipede Game",
// Econometrica 60(4), 803-836. We transcribe the standard formalization G(N, m0, m1) (N >= 1
// decision rounds, piles m0 > m1 >= 1): players alternate starting with player one, each choosing
// from {push, take}. If the game ends on round t in {0, ..., N-1} with final mover p, then
// - p played TAKE: p pockets the doubled big pile 2^t * m0 and the opponent p* the doubled small
//   pile 2^t * m1 ("take the larger pile, give the smaller one across the table");
// - p played PUSH on the last round (all N moves were pushes): p receives 2^N * m1 and p*
//   receives 2^N * m0 (the pushed-across doubled piles swap owners).
// Each push doubles both piles (the coin influx is an externality, not a transfer). Example
// instantiation G(4, 4, 1): take-at-round-t pays the taker 2^t * 4 vs 2^t * 1, all-push ends
// (64, 16) for (player one, player two) since player two makes the last push.
//
// SUBSTITUTION NOTE (task scope). The originally requested "Long-Purda bargaining game" could not
// be pinned to a canonical construction after focused literature search (no standard payoff
// table/protocol surfaces under that name in CFR benchmark suites), so this literature-standard
// sequential trust game was substituted per the task's fallback instruction. It provides exactly
// the requested properties: compact alternating-offer-ish multi-stage machine, general-sum
// terminal payoffs, explicit terminality, scripted truth-table testing, and the textbook
// backward-induction equilibrium "first mover takes immediately" (unique SPE under the growth
// condition m0 > 2*m1, which the default configuration satisfies: 4 > 2*1 -- each round, taking
// now strictly dominates pushing and letting the opponent take next).
//
// FOSG NOTES. Perfect-information alternating game (nothing hidden; private observations are
// always empty and infosets coincide with public states). DETERMINISTIC: no chance outcomes.

/**
 * @brief One decision: pocket the big pile (take) or hand both doubled piles across (push).
 */
struct Move {
   bool take = false;

   friend bool operator==(const Move&, const Move&) = default;
   friend bool operator!=(const Move&, const Move&) = default;
};

/// how a game instance ended
enum class TerminalCause : uint8_t { none = 0, taken, exhausted };

/**
 * @brief Configuration of a centipede instance G(rounds, pile_big, pile_small).
 *
 * @param rounds number N of decision rounds (alternating, starting with player one)
 * @param pile_big the larger starting pile m0 (pocketed by a taker)
 * @param pile_small the smaller starting pile m1 (handed to the other player)
 */
struct Config {
   /// hard cap so the uint64 payoff math (2^rounds * piles) cannot overflow
   static constexpr size_t max_rounds = 32;

   size_t rounds = 4;
   uint32_t pile_big = 4;
   uint32_t pile_small = 1;

   Config() = default;
   Config(size_t rounds_, uint32_t pile_big_, uint32_t pile_small_)
       : rounds(rounds_), pile_big(pile_big_), pile_small(pile_small_)
   {
      validate();
   }

   void validate() const
   {
      if(rounds < 1 or rounds > max_rounds) {
         throw std::invalid_argument(
            "centipede supports 1 to " + std::to_string(max_rounds) + " decision rounds (got "
            + std::to_string(rounds) + ")."
         );
      }
      if(pile_small < 1 or pile_big <= pile_small) {
         throw std::invalid_argument(
            "centipede requires pile_big > pile_small >= 1 (got m0=" + std::to_string(pile_big)
            + ", m1=" + std::to_string(pile_small) + ")."
         );
      }
   }

   [[nodiscard]] friend bool operator==(const Config&, const Config&) = default;
};

/**
 * @brief World state of one centipede game.
 *
 * All data lives in fixed-size members so copying (which happens for every edge of the search
 * tree) stays cheap.
 */
class State {
  public:
   explicit State(Config config = {}) : m_config(config) { m_config.validate(); }

   ////////////////////////////////
   /// API: transitions        ///
   ////////////////////////////////

   void apply_action(Move move)
   {
      if(terminal()) {
         throw std::logic_error("centipede: state is terminal; no further moves.");
      }
      if(not is_valid(move)) {
         throw std::invalid_argument(
            "centipede: illegal move (take=" + std::string(move.take ? "true" : "false") + ")."
         );
      }
      if(move.take) {
         record_payoff_for_active(true);
         m_terminal_cause = TerminalCause::taken;
         return;
      }
      // push: piles double across the table; either the game continues or the last push
      // terminates it with the swapped-pile outcome
      ++m_round;
      if(m_round >= m_config.rounds) {
         record_payoff_for_active(false);
         m_terminal_cause = TerminalCause::exhausted;
         return;
      }
      m_active = opponent(m_active);
   }

   [[nodiscard]] bool is_valid(Move) const { return not terminal(); }

   /// legal moves of whoever is due (empty when terminal or not the acting player)
   [[nodiscard]] std::vector< Move > actions(Player player) const
   {
      std::vector< Move > out;
      if(terminal() or m_active != player) {
         return out;
      }
      out.push_back(Move{true});
      out.push_back(Move{false});
      return out;
   }

   ////////////////////////////////
   /// API: queries            ///
   ////////////////////////////////

   [[nodiscard]] bool terminal() const { return m_terminal_cause != TerminalCause::none; }
   [[nodiscard]] TerminalCause terminal_cause() const { return m_terminal_cause; }

   /**
    * @brief GENERAL-SUM terminal payoffs in doubled-pile units.
    *
    * A taker pockets 2^t * m0 while the opponent collects 2^t * m1; an exhausted game (all pushes)
    * awards 2^N * m0 to the player OPPOSITE the final pusher and 2^N * m1 to the pusher. 0 before
    * terminality. Payoffs deliberately do NOT sum to a constant.
    */
   [[nodiscard]] double payoff(Player player) const
   {
      if(not terminal() or player == Player::none) {
         return 0.;
      }
      return double(m_coins.at(as_int(player)));
   }

   [[nodiscard]] std::array< double, 2 > payoffs() const
   {
      return {payoff(Player::one), payoff(Player::two)};
   }

   /// coins the CURRENT mover pockets by taking right now ({mover_share, opponent_share})
   [[nodiscard]] std::pair< uint64_t, uint64_t > take_resolution() const
   {
      return {take_pow2(m_round) * m_config.pile_big, take_pow2(m_round) * m_config.pile_small};
   }

   /// coins the CURRENT mover ends up with if he pushes and the opponent takes next round
   /// ({mover_share, opponent_share}); this is exactly the backward-induction comparison quantity
   /// against which take_resolution().first must dominate for immediate taking to be optimal
   [[nodiscard]] std::pair< uint64_t, uint64_t > push_then_opponent_takes_resolution() const
   {
      return {
         take_pow2(m_round + 1) * m_config.pile_small, take_pow2(m_round + 1) * m_config.pile_big};
   }

   /// the terminal coin holding of `player` (0 before terminality)
   [[nodiscard]] uint64_t coin_holdings(Player player) const { return m_coins.at(as_int(player)); }

   [[nodiscard]] Player active_player() const { return m_active; }
   [[nodiscard]] const Config& config() const { return m_config; }
   /// current decision round index in [0, N)
   [[nodiscard]] size_t round() const { return m_round; }
   /// the player who pockets the big pile under the hypothetical end-of-game resolution `take`;
   /// exposed so tests never re-derive the owner-swap rule themselves
   [[nodiscard]] Player mover() const { return m_active; }

   [[nodiscard]] bool operator==(const State& other) const
   {
      return m_config == other.m_config && m_active == other.m_active && m_round == other.m_round
             && m_terminal_cause == other.m_terminal_cause && m_coins == other.m_coins;
   }
   [[nodiscard]] bool operator!=(const State& other) const { return not (*this == other); }

  private:
   /////////////////////////////////
   /// API: private utilities   ///
   /////////////////////////////////

   static constexpr uint64_t take_pow2(size_t exponent) { return uint64_t(1) << exponent; }

   /// resolves the CURRENT decision of the acting player into terminal coin holdings
   void record_payoff_for_active(bool take)
   {
      const auto [mover_share, opp_share] =
         take ? std::
               pair{take_pow2(m_round) * m_config.pile_big, take_pow2(m_round) * m_config.pile_small}
              : std::pair{
                 take_pow2(m_config.rounds) * m_config.pile_small,
                 take_pow2(m_config.rounds) * m_config.pile_big};
      m_coins.at(as_int(m_active)) = mover_share;
      m_coins.at(as_int(opponent(m_active))) = opp_share;
      m_active = Player::none;
   }

   /////////////////////////////////
   /// API: data members        ///
   /////////////////////////////////

   Config m_config{};
   Player m_active = Player::one;
   size_t m_round = 0;
   TerminalCause m_terminal_cause = TerminalCause::none;
   std::array< uint64_t, 2 > m_coins{};
};

}  // namespace centipede

#endif  // NOR_CENTIPEDE_STATE_HPP

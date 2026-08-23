
#ifndef NOR_OSHI_ZUMO_STATE_HPP
#define NOR_OSHI_ZUMO_STATE_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <vector>

namespace oshi_zumo {

/// 'none' marks the absence of an acting player (terminal states); its value mirrors
/// nor::Player::unknown so the FOSG adapter's cast stays consistent (cf. pursuit_evasion).
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
// rule transcription notes (OpenSpiel master open_spiel/games/oshi_zumo/oshi_zumo.cc)
// #####################################################################################################################
//
// BOARD. OpenSpiel tracks positions 0 .. 2*size+2 where 0 and 2*size+2 are "off the edge"
// (constructor comment, oshi_zumo.cc:63) and the wrestler spawns on size+1 (oshi_zumo.cc:66).
// We use the equivalent un-shifted board 0 .. 2*size (2*size+1 squares, middle = size, cf. the
// classic description); reaching 0 or 2*size here happens at exactly the same move as reaching
// 0 resp. 2*size+2 there, so the game trees coincide modulo the +1 shift.
//
// PUSH DIRECTION. OpenSpiel: actions[0] > actions[1] => wrestler_pos_++ (oshi_zumo.cc:74f.) and
// arrival at the top boundary crowns player 0 (oshi_zumo.cc:86f.). Identifying player 0 with
// Player::one this reads: the higher bidder pushes the wrestler TOWARDS HIS OWN winning edge --
// equivalently, towards the LOWER bidder's side, whose owner LOSES upon arrival. We follow the
// latter (classic) phrasing: Player::one owns side 0, Player::two owns side 2*size; a strictly
// higher bid pushes the wrestler one step towards the outbid player's side, and the player onto
// whose side the wrestler arrived loses (+1 for the pusher).
//
// PAYMENT RULE (documented DEVIATION from OpenSpiel). OpenSpiel deducts BOTH bids
// unconditionally every round (oshi_zumo.cc:80-83 "// Remove coins. coins_[0] -= actions[0];
// coins_[1] -= actions[1];") including ties. The task specification (and the classic game
// description) instead mandate: ONLY the strictly higher bidder spends his bid; the loser pays
// nothing; on a tie neither player pays and the wrestler stays (no movement on ties matches
// OpenSpiel's strict-inequality movement test, oshi_zumo.cc:73-77). We implement the classic
// rule.
//
// TERMINATION. OpenSpiel IsTerminal() (oshi_zumo.cc:123-126) ends the game on (a) edge arrival,
// (b) total_moves_ >= horizon_, or (c) both purses empty simultaneously. All three clauses are
// transcribed; (c) is unreachable under the winner-pays rule unless both players dump their last
// coins as forced all-in bids or matching drains, but is kept for faithfulness.
//
// HORIZON RESOLUTION (documented DEVIATION from OpenSpiel). OpenSpiel resolves a non-edge
// terminal by comparing the wrestler position against the centre square (oshi_zumo.cc:136-146,
// Returns(); 'alesia'=true scores everything 0 instead). The task specification mandates the
// coin-count rule: at the horizon the player with MORE coins remaining wins, equal coins => draw.
// We implement the coin-count rule for both horizon and both-broke endings; the wrestler position
// itself never breaks a tie. The optional 'alesia' scoring variant is therefore not represented.
//
// HORIZON SEMANTICS. OpenSpiel counts total_moves_ once per SIMULTANEOUS round (increment inside
// DoApplyActions, oshi_zumo.cc:89), i.e. horizon_ bounds joint rounds (= "k moves each"). Its
// default is the constant 1000 (kDefaultHorizon, oshi_zumo.cc:26); per the task we default to
// 3*size and keep the value configurable.
//
// BID LEGALITY. Transcribed from LegalActions (oshi_zumo.cc:96-110): bids run over
// [min_bid, purse]; if the purse cannot cover min_bid the player is FORCED to bid whatever he has
// left (movelist.empty() => movelist.push_back(coins_[player])), which may legitimately be less
// than min_bid (even 0). With the OpenSpiel default min_bid = 0 bidding 0 is always available and
// acts as a pass (legal-set never empty).

/**
 * @brief One simultaneous bid: the number of coins offered this round.
 *
 * Legal amounts are computed from the bidder's purse and the configured minimum bid (see the
 * transcription notes above); the sentinel-free plain amount doubles as pass when 0 is legal.
 */
struct Bid {
   uint32_t amount = 0;

   friend bool operator==(const Bid&, const Bid&) = default;
   friend bool operator!=(const Bid&, const Bid&) = default;
};

/**
 * @brief The phase of a round.
 *
 * The simultaneous bids are sequentialized commit-commit-resolve (repo convention, cf.
 * goofspiel/pursuit_evasion): CommitP1 -> CommitP2 -> Resolve. Because this environment is
 * DETERMINISTIC (Stochasticity::deterministic, monostate chance outcome) the resolve is a single
 * degenerate outcome and is fused into the application of player two's commitment: applying the
 * second Bid applies both bids simultaneously at once. This yields the identical game tree minus
 * a trivial one-outcome node while keeping every non-terminal state attributable to a real acting
 * player (which deterministic-env CFR traversal requires).
 */
enum class Phase : uint8_t { commit_p1 = 0, commit_p2 = 1 };

/// how a game instance ended
enum class TerminalCause : uint8_t { none = 0, edge_arrival, horizon, both_broke };

/**
 * @brief Configuration of an Oshi-Zumo instance.
 *
 * Defaults transcribe OpenSpiel's parameters (kDefaultSize = 3, kDefaultCoins = 50,
 * kDefaultMinBid = 0; oshi_zumo.cc:24-28) except the horizon, which OpenSpiel fixes at the
 * constant kDefaultHorizon = 1000 (oshi_zumo.cc:25) and which the task pins to 3*size.
 *
 * @param size half-width of the board; positions run 0 .. 2*size with the wrestler starting on
 * size
 * @param coins each player's starting purse
 * @param min_bid lowest legal bid (0 enables free passes); OpenSpiel requires
 * 0 <= min_bid <= coins (oshi_zumo.cc:167-171)
 * @param horizon number of joint bid rounds before the coin-count showdown
 */
struct Config {
   /// hard caps so the fixed-size round log and the int16 position bookkeeping stay bounded
   static constexpr size_t max_size = 30;
   static constexpr size_t max_horizon = 64;

   size_t size = 3;
   uint32_t coins = 50;
   uint32_t min_bid = 0;
   size_t horizon = 3 * size;

   Config() = default;
   Config(size_t size_, uint32_t coins_, uint32_t min_bid_)
       : size(size_), coins(coins_), min_bid(min_bid_), horizon(3 * size_)
   {
      validate();
   }
   Config(size_t size_, uint32_t coins_, uint32_t min_bid_, size_t horizon_)
       : size(size_), coins(coins_), min_bid(min_bid_), horizon(horizon_)
   {
      validate();
   }

   void validate() const
   {
      if(size < 1 or size > max_size) {
         throw std::invalid_argument(
            "oshi zumo supports boards of half-width 1 to " + std::to_string(max_size) + " (got "
            + std::to_string(size) + ")."
         );
      }
      if(min_bid > coins) {
         throw std::invalid_argument(
            "oshi zumo requires min_bid <= starting coins (got min_bid=" + std::to_string(min_bid)
            + ", coins=" + std::to_string(coins) + ")."
         );
      }
      if(horizon < 1 or horizon > max_horizon) {
         throw std::invalid_argument(
            "oshi zumo supports horizons of 1 to " + std::to_string(max_horizon) + " rounds (got "
            + std::to_string(horizon) + ")."
         );
      }
   }

   [[nodiscard]] friend bool operator==(const Config&, const Config&) = default;
};

/// the chronological record of one resolved round (history reconstruction + debugging)
struct RoundRecord {
   uint32_t bid_one = 0;
   uint32_t bid_two = 0;
   int16_t wrestler_pos_after = 0;  //< post-resolve position
   bool one_paid = false;  //< true <=> player one won the bid and spent it
   bool two_paid = false;
   TerminalCause cause = TerminalCause::none;  //< non-none only in the final record

   friend bool operator==(const RoundRecord&, const RoundRecord&) = default;
   friend bool operator!=(const RoundRecord&, const RoundRecord&) = default;
};

/**
 * @brief The world state of one deterministic Oshi-Zumo game.
 *
 * Positions live in [0, 2*size] with the wrestler starting on size; sides: 0 belongs to
 * Player::one, 2*size to Player::two. All data lives in fixed-size members so copying (which
 * happens for every edge of the search tree) stays cheap.
 */
class State {
  public:
   explicit State(Config config = {}) : m_config(config)
   {
      m_config.validate();
      m_wrestler_pos = int16_t(m_config.size);
      m_coins = {m_config.coins, m_config.coins};
   }

   ////////////////////////////////
   /// API: transitions        ///
   ////////////////////////////////

   /**
    * Commits a bid for whoever is due. During CommitP1 nothing but the commitment is stored; the
    * joint resolution happens atomically inside the CommitP2 application (fused deterministic
    * resolve, see Phase).
    */
   void apply_action(Bid bid)
   {
      if(terminal()) {
         throw std::logic_error("oshi zumo state is terminal; no further bids.");
      }
      switch(m_phase) {
         case Phase::commit_p1: {
            if(not is_valid(Player::one, bid)) {
               throw std::invalid_argument(
                  "oshi zumo: illegal player-one bid " + std::to_string(bid.amount) + "."
               );
            }
            m_committed[as_int(Player::one)] = bid;
            m_phase = Phase::commit_p2;
            m_active = Player::two;
            return;
         }
         case Phase::commit_p2: {
            if(not is_valid(Player::two, bid)) {
               throw std::invalid_argument(
                  "oshi zumo: illegal player-two bid " + std::to_string(bid.amount) + "."
               );
            }
            _resolve(bid);
            return;
         }
      }
      throw std::logic_error("oshi zumo: unreachable phase.");
   }

   [[nodiscard]] bool is_valid(Bid bid) const
   {
      if(terminal()) {
         return false;
      }
      switch(m_phase) {
         case Phase::commit_p1: return is_valid(Player::one, bid);
         case Phase::commit_p2: return is_valid(Player::two, bid);
      }
      return false;
   }

   /// legality of `bid` for `player` independent of the current phase (OpenSpiel LegalActions)
   [[nodiscard]] bool is_valid(Player player, Bid bid) const
   {
      const auto purse = m_coins.at(as_int(player));
      const auto min = m_config.min_bid;
      if(purse >= min) {
         return bid.amount >= min && bid.amount <= purse;
      }
      // purse below the minimum bid: forced to shove whatever remains across the table
      return bid.amount == purse;
   }

   /// legal bids of whoever is due to commit (empty outside the commit phases / when terminal)
   [[nodiscard]] std::vector< Bid > actions(Player player) const
   {
      std::vector< Bid > out;
      if(terminal() or m_active != player) {
         return out;
      }
      const auto purse = m_coins.at(as_int(player));
      const auto min = m_config.min_bid;
      if(purse >= min) {
         out.reserve(size_t(purse - min) + 1);
         for(uint32_t b : std::views::iota(min, purse + 1)) {
            out.push_back(Bid{b});
         }
      } else {
         // forced all-in fallback (OpenSpiel oshi_zumo.cc:104-109)
         out.push_back(Bid{purse});
      }
      return out;
   }

   ////////////////////////////////
   /// API: queries            ///
   ////////////////////////////////

   [[nodiscard]] bool terminal() const { return m_terminal_cause != TerminalCause::none; }

   [[nodiscard]] TerminalCause terminal_cause() const { return m_terminal_cause; }

   /// the player onto whose side the wrestler arrived (edge arrivals only; nullopt otherwise)
   [[nodiscard]] std::optional< Player > edge_arrival_loser() const
   {
      if(m_terminal_cause != TerminalCause::edge_arrival) {
         return std::nullopt;
      }
      return m_wrestler_pos == 0 ? std::optional{Player::one} : std::optional{Player::two};
   }

   /**
    * @brief Zero-sum terminal payoffs in {-1, 0, +1}.
    *
    * Edge arrival: the player onto whose side the wrestler arrived loses. Horizon / both-broke:
    * more coins wins, equal coins draws (task-mandated rule replacing OpenSpiel's centre-compare,
    * see the transcription notes). 0 before terminality.
    */
   [[nodiscard]] double payoff(Player player) const
   {
      if(not terminal()) {
         return 0.;
      }
      double one_value = [&] {
         switch(m_terminal_cause) {
            case TerminalCause::edge_arrival: return m_wrestler_pos == 0 ? -1. : 1.;
            case TerminalCause::horizon:
            case TerminalCause::both_broke: {
               if(m_coins[as_int(Player::one)] > m_coins[as_int(Player::two)])
                  return 1.;
               if(m_coins[as_int(Player::one)] < m_coins[as_int(Player::two)])
                  return -1.;
               return 0.;
            }
            case TerminalCause::none: return 0.;
         }
         return 0.;
      }();
      return player == Player::one ? one_value : -one_value;
   }

   [[nodiscard]] std::array< double, 2 > payoffs() const
   {
      return {payoff(Player::one), payoff(Player::two)};
   }

   [[nodiscard]] Player active_player() const { return m_active; }
   [[nodiscard]] const Config& config() const { return m_config; }
   [[nodiscard]] Phase phase() const { return m_phase; }
   /// number of fully resolved rounds
   [[nodiscard]] size_t round() const { return m_round; }
   [[nodiscard]] int16_t wrestler_pos() const { return m_wrestler_pos; }
   [[nodiscard]] uint32_t coins(Player player) const { return m_coins.at(as_int(player)); }
   [[nodiscard]] const std::array< uint32_t, 2 >& coins() const { return m_coins; }
   [[nodiscard]] std::optional< Bid > committed_bid(Player player) const
   {
      return m_committed.at(as_int(player));
   }
   /// the per-round records; entries at indices >= round() are not finalized yet
   [[nodiscard]] const std::array< RoundRecord, Config::max_horizon >& rounds() const
   {
      return m_rounds;
   }

   [[nodiscard]] bool operator==(const State& other) const
   {
      return m_config == other.m_config && m_active == other.m_active && m_phase == other.m_phase
             && m_wrestler_pos == other.m_wrestler_pos && m_coins == other.m_coins
             && m_round == other.m_round && m_terminal_cause == other.m_terminal_cause
             && m_committed == other.m_committed && m_rounds == other.m_rounds;
   }
   [[nodiscard]] bool operator!=(const State& other) const { return not (*this == other); }

  private:
   /////////////////////////////////
   /// API: private utilities   ///
   /////////////////////////////////

   /**
    * Fused simultaneous resolution of both commitments: determine the strict winner, let him pay
    * his own bid and push the wrestler one step towards the outbid side (ties: nobody pays, no
    * movement), then evaluate the termination clauses in the precedence edge-arrival >
    * horizon > both-broke (matching OpenSpiel's IsTerminal disjunction order).
    */
   void _resolve(Bid bid_two)
   {
      const auto bid_one = *m_committed.at(as_int(Player::one));

      auto& record = m_rounds.at(m_round);
      record.bid_one = bid_one.amount;
      record.bid_two = bid_two.amount;

      if(bid_one.amount > bid_two.amount) {
         record.one_paid = true;
         m_coins.at(as_int(Player::one)) -= bid_one.amount;
         m_wrestler_pos += 1;  // pushed towards player two's side (position 2*size)
      } else if(bid_two.amount > bid_one.amount) {
         record.two_paid = true;
         m_coins.at(as_int(Player::two)) -= bid_two.amount;
         m_wrestler_pos -= 1;  // pushed towards player one's side (position 0)
      }
      // ties: no payment, no movement
      record.wrestler_pos_after = m_wrestler_pos;

      m_round += 1;
      if(m_wrestler_pos == 0 || m_wrestler_pos == 2 * int16_t(m_config.size)) {
         record.cause = m_terminal_cause = TerminalCause::edge_arrival;
      } else if(m_round >= m_config.horizon) {
         record.cause = m_terminal_cause = TerminalCause::horizon;
      } else if(m_coins[as_int(Player::one)] == 0 && m_coins[as_int(Player::two)] == 0) {
         record.cause = m_terminal_cause = TerminalCause::both_broke;
      } else {
         m_committed = {};
         m_phase = Phase::commit_p1;
         m_active = Player::one;
         return;
      }
      m_active = Player::none;
   }

   /////////////////////////////////
   /// API: data members        ///
   /////////////////////////////////

   Config m_config{};
   Player m_active = Player::one;
   Phase m_phase = Phase::commit_p1;
   int16_t m_wrestler_pos = 0;
   std::array< uint32_t, 2 > m_coins{};
   size_t m_round = 0;
   TerminalCause m_terminal_cause = TerminalCause::none;
   std::array< std::optional< Bid >, 2 > m_committed{};
   std::array< RoundRecord, Config::max_horizon > m_rounds{};
};

}  // namespace oshi_zumo

#endif  // NOR_OSHI_ZUMO_STATE_HPP

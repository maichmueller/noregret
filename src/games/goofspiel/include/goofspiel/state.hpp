
#ifndef NOR_GOOFSPIEL_STATE_HPP
#define NOR_GOOFSPIEL_STATE_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

namespace goofspiel {

/// cards are valued 1..k and tracked through uint16 bitmasks, hence k is capped at 15
constexpr size_t max_deck_size = 15;

enum class Player { chance = -1, one = 0, two = 1 };

template < std::integral To = size_t, typename T >
inline To as_int(T p)
{
   // we let things silently fail in the call site if Player::chance is passed in here for example
   return static_cast< To >(p);
}

/// the bitmask with only bit 'value' set (card values run from 1 upwards)
constexpr uint16_t card_bit(uint8_t value)
{
   return uint16_t(1) << value;
}

/**
 * @brief A player's bid: one of its still unplayed hand cards.
 */
struct Bid {
   uint8_t card;

   friend bool operator==(const Bid&, const Bid&) = default;
};

/**
 * @brief A chance outcome.
 *
 * During the prize-reveal phase 'value' is a remaining prize card value >= 1 drawn uniformly from
 * the deck. The resolve phase is a deterministic transition which is modeled as a degenerate chance
 * step carrying the sentinel value 0 ("confirm resolve").
 */
struct PrizeCard {
   uint8_t value;

   friend bool operator==(const PrizeCard&, const PrizeCard&) = default;
};

/// the winner of a single round (equal bids discard the prize)
enum class RoundOutcome : uint8_t { p1_wins = 0, tie = 1, p2_wins = 2 };

/**
 * @brief The phase of a round.
 *
 * Each round passes through PrizeReveal -> CommitP1 -> CommitP2 -> Resolve, i.e. the simultaneous
 * bids are sequentialized as commit-commit-reveal (the standard OpenSpiel/LiteEFG turn-based
 * encoding). The second committer's infosets therefore naturally carry an "opponent committed" bit.
 */
enum class Phase : uint8_t { prize_reveal = 0, commit_p1 = 1, commit_p2 = 2, resolve = 3 };

/**
 * @brief Configuration of a goofspiel game.
 *
 * Three decks of 'deck_size' cards valued 1..deck_size exist: one private hand per player and the
 * public prize deck. The prize deck is shuffled by chained uniform chance deals without
 * replacement (probability 1/#remaining at each reveal). With imp_info=false bids are publicly
 * revealed right after each resolve; with imp_info=true only the round outcome (winner/tie) is
 * announced and bids are never revealed.
 */
struct GoofspielConfig {
   /// the number of cards per deck k; papers benchmark k in {3, 4, 5}
   size_t deck_size = 3;
   /// true => limited-information observations (bids never revealed)
   bool imp_info = false;

   void validate() const
   {
      if(deck_size < 1 or deck_size > max_deck_size) {
         throw std::invalid_argument(
            "goofspiel supports decks of 1 to " + std::to_string(max_deck_size) + " cards (got "
            + std::to_string(deck_size) + ")."
         );
      }
   }

   friend bool operator==(const GoofspielConfig&, const GoofspielConfig&) = default;
};

/// the record of one completed or ongoing round used for history reconstruction and debugging
struct RoundRecord {
   int8_t prize = 0;  //< 0 while unrevealed
   int8_t bid_one = 0;  //< 0 while uncommitted
   int8_t bid_two = 0;  //< 0 while uncommitted
   RoundOutcome outcome = RoundOutcome::tie;

   friend bool operator==(const RoundRecord&, const RoundRecord&) = default;
};

/**
 * @brief A world state of one goofspiel game.
 *
 * All variable-size data lives in fixed-size members so that copying (which happens for every edge
 * in the search tree) stays cheap.
 */
class State {
  public:
   explicit State(GoofspielConfig config = {}) : m_config(config)
   {
      m_config.validate();
      uint16_t full_deck = 0;
      for(size_t v = 1; v <= m_config.deck_size; ++v) {
         full_deck |= card_bit(uint8_t(v));
      }
      m_hands = {full_deck, full_deck};
      m_prize_deck = full_deck;
   }

   ///////////////////////////////////
   /// API: transitions & queries ///
   ///////////////////////////////////

   void apply_action(Bid action);
   void apply_action(PrizeCard outcome);

   [[nodiscard]] bool is_valid(Bid action) const;
   [[nodiscard]] bool is_valid(PrizeCard outcome) const;

   [[nodiscard]] bool is_terminal() const { return m_round >= m_config.deck_size; }

   /// legal bids of whoever is due to commit (empty outside the commit phases)
   [[nodiscard]] std::vector< Bid > actions() const;

   /// remaining prize cards during the reveal phase and the sentinel confirm outcome during the
   /// resolve phase; empty otherwise
   [[nodiscard]] std::vector< PrizeCard > chance_actions() const;

   /**
    * @brief probability of 'outcome': 1/#remaining for prize reveals and 1 for the deterministic
    * resolve confirmation.
    */
   [[nodiscard]] double chance_probability(PrizeCard outcome) const;

   /// zero-sum reward: score(player) - score(opponent); 0 before terminality
   [[nodiscard]] double payoff(Player player) const;

   /// zero-sum payoff pair {u(one), u(two)}
   [[nodiscard]] std::array< double, 2 > payoffs() const;

   ////////////////////////
   /// API: accessors  ///
   ////////////////////////

   [[nodiscard]] Player active_player() const { return m_active_player; }
   [[nodiscard]] const auto& config() const { return m_config; }
   [[nodiscard]] size_t deck_size() const { return m_config.deck_size; }
   [[nodiscard]] size_t round() const { return m_round; }  //< current round index in [0, k)
   [[nodiscard]] Phase phase() const { return m_phase; }
   [[nodiscard]] int8_t current_prize() const { return m_current_prize; }
   [[nodiscard]] uint16_t prize_deck_mask() const { return m_prize_deck; }
   [[nodiscard]] uint16_t hand_mask(Player player) const { return m_hands.at(as_int(player)); }
   [[nodiscard]] std::vector< uint8_t > hand_cards(Player player) const
   {
      return _unpack(m_hands.at(as_int(player)));
   }
   [[nodiscard]] int32_t score(Player player) const { return m_scores.at(as_int(player)); }
   [[nodiscard]] std::optional< Bid > committed_bid(Player player) const
   {
      return m_committed.at(as_int(player));
   }
   /// the per-round records; entries at indices >= round() are not finalized yet
   [[nodiscard]] const auto& rounds() const { return m_rounds; }

   /// strong equality over all state fields
   bool operator==(const State& other) const
   {
      return m_config == other.m_config && m_hands == other.m_hands && m_scores == other.m_scores
             && m_round == other.m_round && m_phase == other.m_phase
             && m_current_prize == other.m_current_prize && m_committed == other.m_committed
             && m_prize_deck == other.m_prize_deck && m_active_player == other.m_active_player
             && m_rounds == other.m_rounds;
   }

  private:
   /////////////////////////////////
   /// API: private utilities   ///
   /////////////////////////////////

   static std::vector< uint8_t > _unpack(uint16_t mask)
   {
      std::vector< uint8_t > out;
      for(uint8_t v = 1; v < 16; ++v) {
         if(mask & card_bit(v)) {
            out.emplace_back(v);
         }
      }
      return out;
   }

   /////////////////////////////////
   /// API: data members        ///
   /////////////////////////////////

   GoofspielConfig m_config{};
   std::array< uint16_t, 2 > m_hands{};  //< bit v <=> player still holds card v
   std::array< int32_t, 2 > m_scores{};
   size_t m_round = 0;  //< index of the current round
   Phase m_phase = Phase::prize_reveal;
   int8_t m_current_prize = 0;
   std::array< std::optional< Bid >, 2 > m_committed{};
   uint16_t m_prize_deck{};  //< bit v <=> prize card v not revealed yet
   Player m_active_player = Player::chance;
   std::array< RoundRecord, max_deck_size > m_rounds{};
};

inline void State::apply_action(Bid action)
{
   if(is_terminal()) {
      throw std::logic_error("goofspiel: cannot bid on a terminal state.");
   }
   size_t mover = [&] {
      switch(m_phase) {
         case Phase::commit_p1: return as_int(Player::one);
         case Phase::commit_p2: return as_int(Player::two);
         default:
            throw std::invalid_argument(
               "goofspiel: bids are only legal during the commit phases (got phase "
               + std::to_string(unsigned(m_phase)) + ")."
            );
      }
   }();
   if(not (m_hands[mover] & card_bit(action.card))) {
      throw std::invalid_argument(
         "goofspiel: player does not hold the bid card " + std::to_string(unsigned(action.card))
         + "."
      );
   }
   m_hands[mover] &= ~card_bit(action.card);
   m_committed[mover] = action;
   if(mover == as_int(Player::one)) {
      m_rounds[m_round].bid_one = int8_t(action.card);
      m_phase = Phase::commit_p2;
      m_active_player = Player::two;
   } else {
      m_rounds[m_round].bid_two = int8_t(action.card);
      m_phase = Phase::resolve;
      m_active_player = Player::chance;
   }
}

inline void State::apply_action(PrizeCard outcome)
{
   if(is_terminal()) {
      throw std::logic_error("goofspiel: cannot deal on a terminal state.");
   }
   switch(m_phase) {
      case Phase::prize_reveal: {
         if(outcome.value == 0 or not (m_prize_deck & card_bit(outcome.value))) {
            throw std::invalid_argument(
               "goofspiel: " + std::to_string(unsigned(outcome.value))
               + " is not a remaining prize card."
            );
         }
         m_prize_deck &= ~card_bit(outcome.value);
         m_current_prize = int8_t(outcome.value);
         m_rounds[m_round].prize = m_current_prize;
         m_phase = Phase::commit_p1;
         m_active_player = Player::one;
         break;
      }
      case Phase::resolve: {
         if(outcome.value != 0) {
            throw std::invalid_argument(
               "goofspiel: only the deterministic confirm outcome (0) is legal during the resolve "
               "phase."
            );
         }
         const auto bid_one = m_committed[as_int(Player::one)]->card;
         const auto bid_two = m_committed[as_int(Player::two)]->card;
         auto& record = m_rounds[m_round];
         if(bid_one > bid_two) {
            record.outcome = RoundOutcome::p1_wins;
            m_scores[as_int(Player::one)] += m_current_prize;
         } else if(bid_two > bid_one) {
            record.outcome = RoundOutcome::p2_wins;
            m_scores[as_int(Player::two)] += m_current_prize;
         } else {
            record.outcome = RoundOutcome::tie;  // equal bids discard the prize
         }
         m_round++;
         if(not is_terminal()) {
            m_current_prize = 0;
            m_committed = {};
            m_phase = Phase::prize_reveal;
            m_active_player = Player::chance;
         } else {
            // the game is over; no player will be queried on a terminal state
            m_active_player = Player::chance;
         }
         break;
      }
      default:
         throw std::invalid_argument(
            "goofspiel: prize outcomes are only legal during the reveal/resolve phases."
         );
   }
}

inline bool State::is_valid(Bid action) const
{
   if(is_terminal()) {
      return false;
   }
   size_t mover = -1;
   switch(m_phase) {
      case Phase::commit_p1: mover = as_int(Player::one); break;
      case Phase::commit_p2: mover = as_int(Player::two); break;
      default: return false;
   }
   return (m_hands[mover] & card_bit(action.card)) != 0;
}

inline bool State::is_valid(PrizeCard outcome) const
{
   if(is_terminal()) {
      return false;
   }
   switch(m_phase) {
      case Phase::prize_reveal:
         return outcome.value != 0 and (m_prize_deck & card_bit(outcome.value)) != 0;
      case Phase::resolve: return outcome.value == 0;
      default: return false;
   }
}

inline std::vector< Bid > State::actions() const
{
   std::vector< Bid > out;
   switch(m_phase) {
      case Phase::commit_p1: {
         for(uint8_t v = 1; v < 16; ++v) {
            if(m_hands[as_int(Player::one)] & card_bit(v)) {
               out.emplace_back(Bid{v});
            }
         }
         break;
      }
      case Phase::commit_p2: {
         for(uint8_t v = 1; v < 16; ++v) {
            if(m_hands[as_int(Player::two)] & card_bit(v)) {
               out.emplace_back(Bid{v});
            }
         }
         break;
      }
      default: break;
   }
   return out;
}

inline std::vector< PrizeCard > State::chance_actions() const
{
   if(is_terminal()) {
      return {};
   }
   switch(m_phase) {
      case Phase::prize_reveal: {
         std::vector< PrizeCard > out;
         for(uint8_t v = 1; v < 16; ++v) {
            if(m_prize_deck & card_bit(v)) {
               out.emplace_back(PrizeCard{v});
            }
         }
         return out;
      }
      case Phase::resolve: return {PrizeCard{0}};
      default: return {};
   }
}

inline double State::chance_probability(PrizeCard outcome) const
{
   if(is_terminal()) {
      return 0.;
   }
   switch(m_phase) {
      case Phase::prize_reveal: {
         size_t remaining = 0;
         for(uint8_t v = 1; v < 16; ++v) {
            remaining += ((m_prize_deck & card_bit(v)) != 0);
         }
         return remaining ? 1. / double(remaining) : 0.;
      }
      case Phase::resolve: return outcome.value == 0 ? 1. : 0.;
      default: return 0.;
   }
}

inline double State::payoff(Player player) const
{
   if(not is_terminal()) {
      return 0.;
   }
   const double diff = double(m_scores[as_int(Player::one)] - m_scores[as_int(Player::two)]);
   return player == Player::one ? diff : -diff;
}

inline std::array< double, 2 > State::payoffs() const
{
   return {payoff(Player::one), payoff(Player::two)};
}

}  // namespace goofspiel

#endif  // NOR_GOOFSPIEL_STATE_HPP

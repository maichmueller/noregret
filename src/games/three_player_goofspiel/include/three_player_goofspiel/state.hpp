
#ifndef NOR_THREE_PLAYER_GOOFSPIEL_STATE_HPP
#define NOR_THREE_PLAYER_GOOFSPIEL_STATE_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

namespace three_player_goofspiel {

/// cards are valued 1..k and tracked through uint16 bitmasks, hence k is capped at 15
constexpr size_t max_deck_size = 8;

enum class Player { chance = -1, alex = 0, bob = 1, cedric = 2 };

template < std::integral To = size_t, typename T >
inline constexpr To as_int(T p)
{
   // we let things silently fail in the call site if Player::chance is passed in here
   return static_cast< To >(p);
}

/// seat index helpers; team members are alex (seat 0) and bob (seat 1), the adversarial
/// opponent is cedric (seat 2)
constexpr size_t team_member_count = 2;
constexpr size_t player_seat_count = 3;

constexpr bool is_team_member(Player p)
{
   return p == Player::alex or p == Player::bob;
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
 * - prize reveal: 'value' is a remaining prize card value >= 1 drawn uniformly from the deck.
 * - resolve: the deterministic transition modeled as a degenerate chance step carrying the
 *   sentinel value 0 and kind confirm ("confirm resolve").
 * - deal: only in split-deck mode as the very first event; encodes alex's dealt half-mask,
 *   with bob receiving the complementary half of the full deck.
 */
struct ChanceOutcome {
   enum class Kind : uint8_t { prize = 0, confirm = 1, deal = 2 };
   Kind kind = Kind::prize;
   uint8_t value = 0;  //< prize card value (prize/resolve kinds)
   uint16_t mask_a = 0;  //< alex's dealt subdeck mask (deal kind)

   friend bool operator==(const ChanceOutcome&, const ChanceOutcome&) = default;
};

/// winner of a single round among the three bidders (ties at the maximum discard the prize)
enum class RoundWinner : uint8_t { alex = 0, bob = 1, cedric = 2, tie = 3 };

/**
 * @brief The phase of a round.
 *
 * Each round passes through PrizeReveal -> CommitAlex -> CommitBob -> CommitCedric -> Resolve,
 * i.e. the three simultaneous bids are sequentialized as commit-commit-commit-reveal (the
 * standard turn-based encoding). The second/third committers' infosets therefore naturally
 * carry "opponent committed" bits while bid VALUES stay hidden until the resolve.
 */
enum class Phase : uint8_t {
   deal = 0,
   prize_reveal = 1,
   commit_alex = 2,
   commit_bob = 3,
   commit_cedric = 4,
   resolve = 5
};

/**
 * @brief Configuration of a three-player goofspiel game.
 *
 * Team members are alex and bob acting independently against the adversarial cedric -- no
 * intra-team communication during play. Scoring is zero-sum with an EQUAL SPLIT of the team's
 * accumulated score between the members (the payoff-split rule of the TB-DAG papers'
 * experiments): every team member receives 0.5*(team_score - cedric_score), and cedric
 * receives the negation of the sum of those team-member payoffs.
 *
 * Two deal modes exist:
 * - identical (classic goofspiel): everyone holds the full deck 1..k. The team then has NO
 *   uncommon information and the TB-DAG degenerates towards public states.
 * - split_half: chance deals alex a uniformly random floor(k/2)-card subset of the deck
 *   (concealed from everyone else, including his partner bob), bob receives the complementary
 *   half, and cedric keeps the full deck. This introduces genuine uncommon information inside
 *   the team so that team beliefs collapse worlds non-trivially.
 */
struct GoofspielConfig {
   /// the number of cards per deck k
   size_t deck_size = 3;
   /// true => limited-information observations (bids never revealed; only round outcomes are
   /// announced). false => bids of all three players are published at each resolve.
   bool imp_info = false;
   /// true => concealed split-half deal for the team members (see class comment)
   bool split_half_deal = false;

   void validate() const
   {
      if(deck_size < 1 or deck_size > max_deck_size) {
         throw std::invalid_argument(
            "three_player_goofspiel supports decks of 1 to " + std::to_string(max_deck_size)
            + " cards (got " + std::to_string(deck_size) + ")."
         );
      }
      if(split_half_deal and (deck_size % 2 != 0)) {
         throw std::invalid_argument(
            "split_half_deal requires an even deck size (got " + std::to_string(deck_size) + ")."
         );
      }
   }

   friend bool operator==(const GoofspielConfig&, const GoofspielConfig&) = default;
};

/// the record of one completed or ongoing round used for history reconstruction and debugging
struct RoundRecord {
   int8_t prize = 0;  //< 0 while unrevealed
   int8_t bid_alex = 0;  //< 0 while uncommitted
   int8_t bid_bob = 0;  //< 0 while uncommitted
   int8_t bid_cedric = 0;  //< 0 while uncommitted
   RoundWinner outcome = RoundWinner::tie;

   friend bool operator==(const RoundRecord&, const RoundRecord&) = default;
};

/**
 * @brief A world state of one three-player goofspiel game.
 *
 * All variable-size data lives in fixed-size members so that copying (which happens for every
 * edge in the search tree) stays cheap.
 */
class State {
  public:
   explicit State(GoofspielConfig config = {}) : m_config(config)
   {
      m_config.validate();
      m_hands = {_full_deck(), _full_deck(), _full_deck()};
      m_prize_deck = _full_deck();
      if(m_config.split_half_deal) {
         m_active_player = Player::chance;
         m_phase = Phase::deal;
      } else {
         m_active_player = Player::chance;
         m_phase = Phase::prize_reveal;
      }
   }

   ///////////////////////////////////
   /// API: transitions & queries ///
   ///////////////////////////////////

   void apply_action(Bid action);
   void apply_action(ChanceOutcome outcome);

   [[nodiscard]] bool is_valid(Bid action) const;
   [[nodiscard]] bool is_valid(ChanceOutcome outcome) const;

   [[nodiscard]] bool is_terminal() const { return m_round >= m_config.deck_size; }

   /// legal bids of whoever is due to commit (empty outside the commit phases)
   [[nodiscard]] std::vector< Bid > actions() const;

   /// remaining prize cards during the reveal phase, the sentinel confirm outcome during the
   /// resolve phase, and all legal deals during the initial deal phase (split mode); empty
   /// otherwise
   [[nodiscard]] std::vector< ChanceOutcome > chance_actions() const;

   /**
    * @brief probability of 'outcome': 1/#remaining for prize reveals, 1 for the deterministic
    * resolve confirmation and 1/C(k, k/2) per deal in split mode.
    */
   [[nodiscard]] double chance_probability(ChanceOutcome outcome) const;

   /// zero-sum reward: 0.5*(team score - cedric score) for each team member and the negation of
   /// their sum for cedric; 0 before terminality
   [[nodiscard]] double payoff(Player player) const;

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
   /// which team member was dealt which half (split mode); both empty before/during the deal
   [[nodiscard]] std::pair< uint16_t, uint16_t > dealt_halves() const
   {
      return {m_dealt_alex, m_dealt_bob};
   }

   /// strong equality over all state fields
   bool operator==(const State& other) const
   {
      return m_config == other.m_config && m_hands == other.m_hands && m_scores == other.m_scores
             && m_round == other.m_round && m_phase == other.m_phase
             && m_current_prize == other.m_current_prize && m_committed == other.m_committed
             && m_prize_deck == other.m_prize_deck && m_active_player == other.m_active_player
             && m_dealt_alex == other.m_dealt_alex && m_dealt_bob == other.m_dealt_bob
             && m_rounds == other.m_rounds;
   }

  private:
   /////////////////////////////////
   /// API: private utilities   ///
   /////////////////////////////////

   [[nodiscard]] uint16_t _full_deck() const
   {
      uint16_t full_deck = 0;
      for(size_t v = 1; v <= m_config.deck_size; ++v) {
         full_deck |= card_bit(uint8_t(v));
      }
      return full_deck;
   }

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

   static size_t _hand_count(uint16_t mask)
   {
      size_t n = 0;
      for(; mask != 0; mask &= (mask - 1)) {
         ++n;
      }
      return n;
   }

   /////////////////////////////////
   /// API: data members        ///
   /////////////////////////////////

   GoofspielConfig m_config{};
   std::array< uint16_t, player_seat_count > m_hands{};  //< bit v <=> player still holds card v
   std::array< int32_t, player_seat_count > m_scores{};
   size_t m_round = 0;  //< index of the current round
   Phase m_phase = Phase::deal;
   int8_t m_current_prize = 0;
   std::array< std::optional< Bid >, player_seat_count > m_committed{};
   uint16_t m_prize_deck{};  //< bit v <=> prize card v not revealed yet
   uint16_t m_dealt_alex = 0;  //< alex's dealt half (split mode), 0 otherwise/unrevealed
   uint16_t m_dealt_bob = 0;  //< bob's dealt complement half (split mode), 0 otherwise
   Player m_active_player = Player::chance;
   std::array< RoundRecord, max_deck_size > m_rounds{};
};

inline void State::apply_action(Bid action)
{
   if(is_terminal()) {
      throw std::logic_error("three_player_goofspiel: cannot bid on a terminal state.");
   }
   size_t mover = [&] {
      switch(m_phase) {
         case Phase::commit_alex: return as_int(Player::alex);
         case Phase::commit_bob: return as_int(Player::bob);
         case Phase::commit_cedric: return as_int(Player::cedric);
         default:
            throw std::invalid_argument(
               "three_player_goofspiel: bids are only legal during the commit phases (got phase "
               + std::to_string(unsigned(m_phase)) + ")."
            );
      }
   }();
   if(not (m_hands[mover] & card_bit(action.card))) {
      throw std::invalid_argument(
         "three_player_goofspiel: player does not hold the bid card "
         + std::to_string(unsigned(action.card)) + "."
      );
   }
   m_hands[mover] &= ~card_bit(action.card);
   m_committed[mover] = action;
   switch(mover) {
      case as_int(Player::alex): {
         m_rounds[m_round].bid_alex = int8_t(action.card);
         m_phase = Phase::commit_bob;
         m_active_player = Player::bob;
         break;
      }
      case as_int(Player::bob): {
         m_rounds[m_round].bid_bob = int8_t(action.card);
         m_phase = Phase::commit_cedric;
         m_active_player = Player::cedric;
         break;
      }
      default: {
         m_rounds[m_round].bid_cedric = int8_t(action.card);
         m_phase = Phase::resolve;
         m_active_player = Player::chance;
         break;
      }
   }
}

inline void State::apply_action(ChanceOutcome outcome)
{
   if(is_terminal()) {
      throw std::logic_error("three_player_goofspiel: cannot deal on a terminal state.");
   }
   switch(outcome.kind) {
      case ChanceOutcome::Kind::deal: {
         if(m_phase != Phase::deal) {
            throw std::invalid_argument(
               "three_player_goofspiel: deals are only legal during the initial deal phase."
            );
         }
         const auto half = m_config.deck_size / 2;
         const uint16_t full = _full_deck();
         if(_hand_count(outcome.mask_a) != half
            or (outcome.mask_a & ~full) != 0 /* any bit outside the deck */) {
            throw std::invalid_argument(
               "three_player_goofspiel: illegal deal: mask must select exactly "
               + std::to_string(half) + " cards of the deck."
            );
         }
         m_dealt_alex = outcome.mask_a;
         m_dealt_bob = full & ~outcome.mask_a;
         m_hands[as_int(Player::alex)] = m_dealt_alex;
         m_hands[as_int(Player::bob)] = m_dealt_bob;
         m_phase = Phase::prize_reveal;
         m_active_player = Player::chance;
         break;
      }
      case ChanceOutcome::Kind::prize: {
         if(m_phase != Phase::prize_reveal) {
            throw std::invalid_argument(
               "three_player_goofspiel: prize outcomes are only legal during the reveal phase."
            );
         }
         if(outcome.value == 0 or not (m_prize_deck & card_bit(outcome.value))) {
            throw std::invalid_argument(
               "three_player_goofspiel: " + std::to_string(unsigned(outcome.value))
               + " is not a remaining prize card."
            );
         }
         m_prize_deck &= ~card_bit(outcome.value);
         m_current_prize = int8_t(outcome.value);
         m_rounds[m_round].prize = m_current_prize;
         m_phase = Phase::commit_alex;
         m_active_player = Player::alex;
         break;
      }
      case ChanceOutcome::Kind::confirm: {
         if(m_phase != Phase::resolve) {
            throw std::invalid_argument(
               "three_player_goofspiel: the confirm outcome is only legal during the resolve "
               "phase."
            );
         }
         auto& record = m_rounds[m_round];
         const auto best = std::max({record.bid_alex, record.bid_bob, record.bid_cedric});
         const size_t winners = size_t(record.bid_alex == best) + size_t(record.bid_bob == best)
                                + size_t(record.bid_cedric == best);
         if(winners == 1) {
            if(best == record.bid_alex) {
               record.outcome = RoundWinner::alex;
               m_scores[as_int(Player::alex)] += m_current_prize;
            } else if(best == record.bid_bob) {
               record.outcome = RoundWinner::bob;
               m_scores[as_int(Player::bob)] += m_current_prize;
            } else {
               record.outcome = RoundWinner::cedric;
               m_scores[as_int(Player::cedric)] += m_current_prize;
            }
         } else {
            record.outcome = RoundWinner::tie;  // ties at the max discard the prize
         }
         m_round++;
         m_current_prize = 0;
         m_committed = {};
         if(not is_terminal()) {
            m_phase = Phase::prize_reveal;
         }
         // on terminality no player will be queried anymore
         m_active_player = Player::chance;
         break;
      }
   }
}

inline bool State::is_valid(Bid action) const
{
   if(is_terminal()) {
      return false;
   }
   size_t mover = -1;
   switch(m_phase) {
      case Phase::commit_alex: mover = as_int(Player::alex); break;
      case Phase::commit_bob: mover = as_int(Player::bob); break;
      case Phase::commit_cedric: mover = as_int(Player::cedric); break;
      default: return false;
   }
   return (m_hands[mover] & card_bit(action.card)) != 0;
}

inline bool State::is_valid(ChanceOutcome outcome) const
{
   if(is_terminal()) {
      return false;
   }
   switch(outcome.kind) {
      case ChanceOutcome::Kind::deal: {
         if(m_phase != Phase::deal) {
            return false;
         }
         return _hand_count(outcome.mask_a) == m_config.deck_size / 2
                and (outcome.mask_a & ~_full_deck()) == 0;
      }
      case ChanceOutcome::Kind::prize:
         return m_phase == Phase::prize_reveal and outcome.value != 0
                and (m_prize_deck & card_bit(outcome.value)) != 0;
      case ChanceOutcome::Kind::confirm: return m_phase == Phase::resolve and outcome.value == 0;
   }
   return false;
}

inline std::vector< Bid > State::actions() const
{
   std::vector< Bid > out;
   switch(m_phase) {
      case Phase::commit_alex: {
         for(uint8_t v = 1; v < 16; ++v) {
            if(m_hands[as_int(Player::alex)] & card_bit(v)) {
               out.emplace_back(Bid{v});
            }
         }
         break;
      }
      case Phase::commit_bob: {
         for(uint8_t v = 1; v < 16; ++v) {
            if(m_hands[as_int(Player::bob)] & card_bit(v)) {
               out.emplace_back(Bid{v});
            }
         }
         break;
      }
      case Phase::commit_cedric: {
         for(uint8_t v = 1; v < 16; ++v) {
            if(m_hands[as_int(Player::cedric)] & card_bit(v)) {
               out.emplace_back(Bid{v});
            }
         }
         break;
      }
      default: break;
   }
   return out;
}

inline std::vector< ChanceOutcome > State::chance_actions() const
{
   if(is_terminal()) {
      return {};
   }
   switch(m_phase) {
      case Phase::deal: {
         // uniform over C(k, k/2) subsets held by alex; bob receives the complement
         std::vector< ChanceOutcome > out;
         const size_t half = m_config.deck_size / 2;
         const uint16_t full = _full_deck();
         for(uint32_t sub = full;; sub = (sub - 1) & full) {
            if(_hand_count(uint16_t(sub)) == half) {
               out.emplace_back(ChanceOutcome{
                  .kind = ChanceOutcome::Kind::deal, .value = 0, .mask_a = uint16_t(sub)});
            }
            if(sub == 0) {
               break;
            }
         }
         return out;
      }
      case Phase::prize_reveal: {
         std::vector< ChanceOutcome > out;
         for(uint8_t v = 1; v < 16; ++v) {
            if(m_prize_deck & card_bit(v)) {
               out.emplace_back(ChanceOutcome{
                  .kind = ChanceOutcome::Kind::prize, .value = v, .mask_a = 0});
            }
         }
         return out;
      }
      case Phase::resolve:
         return {ChanceOutcome{.kind = ChanceOutcome::Kind::confirm, .value = 0, .mask_a = 0}};
      default: return {};
   }
}

inline double State::chance_probability(ChanceOutcome outcome) const
{
   if(is_terminal()) {
      return 0.;
   }
   switch(outcome.kind) {
      case ChanceOutcome::Kind::deal: {
         if(m_phase != Phase::deal) {
            return 0.;
         }
         // 1 / C(k, k/2): uniform over the valid halves
         const auto choices = chance_actions();
         for(const auto& choice : choices) {
            if(choice.mask_a == outcome.mask_a) {
               return 1. / double(choices.size());
            }
         }
         return 0.;
      }
      case ChanceOutcome::Kind::prize: {
         if(m_phase != Phase::prize_reveal) {
            return 0.;
         }
         size_t remaining = 0;
         for(uint8_t v = 1; v < 16; ++v) {
            remaining += ((m_prize_deck & card_bit(v)) != 0);
         }
         return remaining ? 1. / double(remaining) : 0.;
      }
      case ChanceOutcome::Kind::confirm: return m_phase == Phase::resolve ? 1. : 0.;
   }
   return 0.;
}

inline double State::payoff(Player player) const
{
   if(not is_terminal()) {
      return 0.;
   }
   const double team_score = double(m_scores[as_int(Player::alex)] + m_scores[as_int(Player::bob)]);
   const double opp_score = double(m_scores[as_int(Player::cedric)]);
   if(is_team_member(player)) {
      // equal-split payoff rule of the TB-DAG papers' experiments
      return 0.5 * (team_score - opp_score);
   }
   return opp_score - team_score;
}

}  // namespace three_player_goofspiel

#endif  // NOR_THREE_PLAYER_GOOFSPIEL_STATE_HPP

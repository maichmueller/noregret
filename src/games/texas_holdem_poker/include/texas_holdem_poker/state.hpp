
#ifndef NOR_TEXAS_HOLDEM_POKER_STATE_HPP
#define NOR_TEXAS_HOLDEM_POKER_STATE_HPP

#include <array>
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <range/v3/all.hpp>
#include <stdexcept>
#include <vector>

#include "common/common.hpp"

namespace texholdem {

/// the maximal number of players this game supports (2 - 6)
constexpr size_t max_player_count = 6;
/// number of hole cards per player
constexpr size_t hole_cards_per_player = 2;
/// number of community cards on a full board (flop + turn + river)
constexpr size_t community_card_count = 5;

enum class Player { chance = -1, one = 0, two = 1, three = 2, four = 3, five = 4, six = 5 };

template < std::integral To = size_t, typename T >
inline To as_int(T p)
{
   // we let things silently fail in the call site if Player::chance is passed in here for example
   return static_cast< To >(p);
}

enum class Rank {
   two = 2,
   three = 3,
   four = 4,
   five = 5,
   six = 6,
   seven = 7,
   eight = 8,
   nine = 9,
   ten = 10,
   jack = 11,
   queen = 12,
   king = 13,
   ace = 14
};

enum class Suit { clubs = 0, diamonds = 1, hearts = 2, spades = 3 };

struct Card {
   Rank rank;
   Suit suit;

   [[nodiscard]] size_t index() const
   {
      // a unique index in [0, 52) into the canonical deck ordering
      return 4 * (as_int(rank) - 2) + as_int(suit);
   }
};

inline bool operator==(const Card& card1, const Card& card2)
{
   return card1.rank == card2.rank && card1.suit == card2.suit;
}
inline bool operator!=(const Card& card1, const Card& card2)
{
   return not (card1 == card2);
}

enum class Street : uint8_t { preflop = 0, flop = 1, turn = 2, river = 3 };

/**
 * @brief The type of a (betting) action.
 *
 * The 'amount' field of Action is interpreted as follows:
 * - fold/check/call/all_in: the amount field is ignored (call resolves the to-call amount
 *   automatically, all_in commits the full remaining stack).
 * - bet/raise: 'amount' is the *target total street contribution* ("raise-to" semantics),
 *   i.e. after applying the action the acting player's street contribution equals 'amount'.
 */
enum class ActionType : uint8_t { fold = 0, check = 1, call = 2, bet = 3, raise = 4, all_in = 5 };

struct Action {
   ActionType kind;
   double amount = 0.;
};

inline bool operator==(const Action& action1, const Action& action2)
{
   return action1.kind == action2.kind && action1.amount == action2.amount;
}
inline bool operator!=(const Action& action1, const Action& action2)
{
   return not (action1 == action2);
}

/// an applied betting action together with the acting player and the street it was played on
struct ActionRecord {
   Player player;
   Action action;
   Street street;
};

inline bool operator==(const ActionRecord& left, const ActionRecord& right)
{
   return left.player == right.player && left.action == right.action && left.street == right.street;
}
inline bool operator!=(const ActionRecord& left, const ActionRecord& right)
{
   return not (left == right);
}

/**
 * @brief Configuration of a no-limit texas hold'em hand.
 *
 * Bet sizing follows a discrete ladder of increments expressed in multiples of the big blind
 * (mirroring leduc_poker's fixed bet sizes) plus an automatic explicit all-in option whenever it
 * adds a genuinely new wager size. This keeps the action space finite (a requirement of tabular
 * CFR) while retaining no-limit min-raise/stack/side-pot legality rules.
 */
struct PokerConfig {
   size_t n_players = 2;  //< number of players, has to be within [2, max_player_count]
   double starting_stack = 200.;  //< stack size of every player at hand start
   double small_blind = 1.;  //< the small blind posting
   double big_blind = 2.;  //< the big blind posting (also the minimal bet size)
   /// bet/raise increments in multiples of the big blind offered on each street
   std::vector< double > bet_size_multiples = {1., 2., 4., 8.};
   /// optional per-player starting stacks (if empty every player receives 'starting_stack')
   std::vector< double > starting_stacks{};

   PokerConfig() = default;
   PokerConfig(
      size_t n_players_,
      double starting_stack_ = 200.,
      double small_blind_ = 1.,
      double big_blind_ = 2.,
      std::vector< double > bet_size_multiples_ = {1., 2., 4., 8.},
      std::vector< double > starting_stacks_ = {}
   )
       : n_players(n_players_),
         starting_stack(starting_stack_),
         small_blind(small_blind_),
         big_blind(big_blind_),
         bet_size_multiples(std::move(bet_size_multiples_)),
         starting_stacks(std::move(starting_stacks_))
   {
   }

   void validate() const
   {
      if(n_players < 2 or n_players > max_player_count) {
         throw std::invalid_argument(
            "texas hold'em supports between 2 and " + std::to_string(max_player_count)
            + " players (got " + std::to_string(n_players) + ")."
         );
      }
      if(big_blind <= 0. or small_blind <= 0. or small_blind > big_blind) {
         throw std::invalid_argument("Blinds have to be positive with small_blind <= big_blind.");
      }
      if(starting_stack < big_blind) {
         throw std::invalid_argument("Starting stack has to be at least one big blind.");
      }
      if(not starting_stacks.empty() && starting_stacks.size() < n_players) {
         throw std::invalid_argument(
            "'starting_stacks' has to provide a stack for each of the " + std::to_string(n_players)
            + " players."
         );
      }
   }

   friend bool operator==(const PokerConfig& left, const PokerConfig& right)
   {
      return left.n_players == right.n_players && left.starting_stack == right.starting_stack
             && left.small_blind == right.small_blind && left.big_blind == right.big_blind
             && left.bet_size_multiples == right.bet_size_multiples
             && left.starting_stacks == right.starting_stacks;
   }
};

/**
 * @brief A world state of a single no-limit texas hold'em hand.
 *
 * All variable-size data is kept in fixed size containers so that copying (which happens for
 * every edge in the search tree) stays cheap. Equality is strong (all fields) and matching
 * std::hash specializations for the value types (Card, Action, ...) are provided in utils.hpp.
 */
class State {
   struct PlayerRecord {
      double stack = 0.;  //< chips still in front of the player
      double street_contribution{};  //< chips committed by the player during the current street
      double total_contribution{};  //< chips committed by the player during the entire hand
                                    //< (including the current street contributions)
      bool folded = false;
      bool allin = false;
      bool acted = false;  //< whether the player acted since the last aggressive action
   };

  public:
   explicit State(PokerConfig config = {});

   ////////////////////////////////////
   /// API: transitions and queries ///
   ////////////////////////////////////

   void apply_action(Action action);
   void apply_action(Card outcome);

   [[nodiscard]] bool is_terminal() const;
   [[nodiscard]] std::vector< Action > actions() const;
   [[nodiscard]] std::vector< Card > chance_actions() const;
   [[nodiscard]] double chance_probability(Card outcome) const;
   [[nodiscard]] bool is_valid(Action action) const;
   [[nodiscard]] bool is_valid(Card outcome) const;
   /// reward of `player` at this state (zero-sum across players at terminal states)
   [[nodiscard]] double payoff(Player player) const;
   [[nodiscard]] std::vector< double > payoffs() const;

   /////////////////////////
   /// API: accessors    ///
   /////////////////////////

   [[nodiscard]] Player active_player() const { return m_active_player; }
   /// all seated players (including folded ones)
   [[nodiscard]] std::vector< Player > players() const;
   /// all players that have not (yet) folded
   [[nodiscard]] std::vector< Player > remaining_players() const;
   [[nodiscard]] Street street() const { return m_street; }
   [[nodiscard]] const auto& config() const { return m_config; }

   [[nodiscard]] auto& hole_cards(Player player) const { return m_holes[as_int(player)]; }
   /// the i-th dealt hole card of `player` (i in [0, 2))
   [[nodiscard]] std::optional< Card > hole_card(Player player, size_t i) const
   {
      return m_holes[as_int(player)][i];
   }
   [[nodiscard]] size_t holes_dealt() const { return m_holes_dealt; }
   /// the player who would receive the next dealt hole card
   [[nodiscard]] Player next_deal_recipient() const
   {
      return Player((m_sb + (m_holes_dealt % m_config.n_players)) % m_config.n_players);
   }
   [[nodiscard]] size_t n_community_cards() const { return m_board_dealt; }
   /// the currently dealt community cards in deal order (size() == n_community_cards())
   [[nodiscard]] std::vector< Card > community_cards() const;
   /// the next community card to be dealt (empty if none pending)
   [[nodiscard]] std::optional< Card > next_community_slot() const
   {
      return m_board_dealt < community_card_count ? m_board[m_board_dealt] : std::nullopt;
   }
   /// number of cards still in the deck
   [[nodiscard]] size_t deck_size() const { return 52 - m_dealt.count(); }
   [[nodiscard]] bool is_dealt(Card card) const { return m_dealt[card.index()]; }

   [[nodiscard]] double stack(Player player) const { return m_players[as_int(player)].stack; }
   [[nodiscard]] double street_contribution(Player player) const
   {
      return m_players[as_int(player)].street_contribution;
   }
   [[nodiscard]] double total_contribution(Player player) const
   {
      return m_players[as_int(player)].total_contribution;
   }
   [[nodiscard]] bool has_folded(Player player) const { return m_players[as_int(player)].folded; }
   [[nodiscard]] bool is_allin(Player player) const { return m_players[as_int(player)].allin; }
   /// all chips currently in the pot(s): street + total contributions of all players
   [[nodiscard]] double pot() const;
   /// the highest street contribution any live player currently has to match
   [[nodiscard]] double current_total_bet() const { return m_current_total_bet; }
   /// the size of the last bet/raise increment (defines the minimum raise)
   [[nodiscard]] double last_increment() const { return m_last_increment; }
   /// the position index of the last aggressor (-1 if none on this street)
   [[nodiscard]] int last_aggressor() const { return m_last_aggressor; }
   [[nodiscard]] int dealer_pos() const { return m_dealer; }
   [[nodiscard]] int small_blind_pos() const { return m_sb; }
   [[nodiscard]] int big_blind_pos() const { return m_bb; }
   /// the public history of betting actions taken so far (in order)
   [[nodiscard]] auto& actions_history() const { return m_actions; }
   /// strong equality over all state fields
   bool operator==(const State& other) const;

  private:
   //////////////////////////////////////
   /// API: private member functions ///
   //////////////////////////////////////

   [[nodiscard]] double _to_call(Player player) const;
   /// smallest legal raise-to target when facing a wager (current bet + minimum raise)
   [[nodiscard]] double _min_raise_to(Player player) const;
   [[nodiscard]] double _max_raise_to(Player player) const;
   [[nodiscard]] bool _can_act(const PlayerRecord& record) const
   {
      return not record.folded and not record.allin;
   }
   /// number of players which may still act voluntarily
   [[nodiscard]] size_t _n_actionable() const;
   /// betting round finished iff every live player is allin or has matched the current wager
   [[nodiscard]] bool _round_complete() const;
   /// first seat >= start whose owner may act
   [[nodiscard]] int _first_actionable_from(int start) const;
   void _start_new_betting_round();
   /// close the current street: sweep contributions, advance street, set up the next round
   void _conclude_street();
   /// commit chips such that the player's street contribution becomes 'target_total'
   void _commit_wager_to(Player player, double target_total);
   /// commit 'delta' chips of the given player
   void _commit_wager_delta(Player player, double delta);
   /// match the current wager (or check if nothing is due); caps at the player's stack
   void _apply_call_or_check(Player player);
   [[nodiscard]] std::vector< Action > _betting_actions(Player player) const;
   /// build the layered side pots from the players' total contributions
   [[nodiscard]] std::vector< std::pair< double, std::vector< Player > > > _build_side_pots() const;
   /// the best 5-card score of the player given hole + community cards
   [[nodiscard]] uint32_t _hand_score(Player player) const;

   ////////////////////////////////
   /// API: private data members ///
   ////////////////////////////////

   PokerConfig m_config;
   std::array< PlayerRecord, max_player_count > m_players{};
   int m_dealer = 0;  //< button position
   int m_sb = 0;  //< small blind position
   int m_bb = 0;  //< big blind position
   Player m_active_player = Player::chance;
   Street m_street = Street::preflop;
   /// bit i marks the card with index i as removed from the deck
   std::bitset< 52 > m_dealt{};
   size_t m_holes_dealt = 0;
   size_t m_board_dealt = 0;
   std::array< std::optional< Card >, community_card_count > m_board{};
   std::array< std::array< std::optional< Card >, hole_cards_per_player >, max_player_count >
      m_holes{};
   double m_current_total_bet = 0.;
   double m_last_increment = 0.;
   int m_last_aggressor = -1;
   std::vector< ActionRecord > m_actions{};
   /// true once no further betting action can take place: either the river betting round
   /// concluded or every remaining player is all-in (the board then runs out to showdown)
   bool m_betting_finished = false;
};

}  // namespace texholdem

#endif  // NOR_TEXAS_HOLDEM_POKER_STATE_HPP

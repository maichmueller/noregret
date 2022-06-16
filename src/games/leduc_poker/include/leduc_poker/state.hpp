
#ifndef NOR_LEDUC_POKER_STATE_HPP
#define NOR_LEDUC_POKER_STATE_HPP

#include <array>
#include <optional>
#include <range/v3/all.hpp>
#include <sstream>
#include <vector>

#include "common/common.hpp"

namespace leduc {

enum class Player {
   chance = -1,
   one = 0,
   two = 1,
   three = 2,
   four = 3,
   five = 4,
   six = 5,
   seven = 6,
   eight = 7,
   nine = 8,
   ten = 9
};

template < typename T, std::integral To = size_t >
inline auto as_integral(T p)
{
   // we let things silently fail in the call site if Player::chance is passed in here for example
   return static_cast< To >(p);
};

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
};

inline bool operator==(const Card& card1, const Card& card2)
{
   return card1.rank == card2.rank and card1.suit == card2.suit;
}

enum class ActionType {
   check = 0,  // check also acts as a call upon betting
   fold = 1,
   bet = 2
};

struct Action {
   ActionType action_type;
   float bet = 0.f;
};

inline bool operator==(const Action& action1, const Action& action2)
{
   // we are never expecting large floating point values here so the absolute comparison should
   // suffice for the use case at hand
   return action1.action_type == action2.action_type
          and std::fabs(action1.bet - action2.bet) <= std::numeric_limits< float >::epsilon();
}

struct LeducConfig {
   size_t n_players = 2;
   size_t n_raises_allowed = 2;
   float small_blind = 1.;
   std::vector< float > bet_sizes_round_one = {2};
   std::vector< float > bet_sizes_round_two = {4};
   std::vector< Card > available_cards = {
      {Rank::jack, Suit::clubs},
      {Rank::jack, Suit::diamonds},
      {Rank::queen, Suit::clubs},
      {Rank::queen, Suit::diamonds},
      {Rank::king, Suit::clubs},
      {Rank::king, Suit::diamonds}};
};

auto leduc5_config()
{
   // returns a bigger betting range config, which will blow up the action space and thus game tree
   // enormously! as per Noam Brown's thesis: Nr of information sets for Leduc: 288, Leduc-5: 34224
   return LeducConfig{
      .bet_sizes_round_one = {0.5, 1, 2, 4, 8}, .bet_sizes_round_two = {1, 2, 4, 8, 16}};
}

/**
 * @brief stores the currently commited action sequence
 */
struct HistorySinceBet {
   HistorySinceBet(std::vector< std::optional< Action > > cont) : container(std::move(cont)) {}
   std::vector< std::optional< Action > > container{};

   auto& operator[](Player player) { return container[as_integral(player)]; }

   auto& operator[](Player player) const { return container[as_integral(player)]; }

   void reset()
   {
      ranges::for_each(container, [](auto& action_opt) { action_opt.reset(); });
   }

   bool all_acted(const std::vector< Player >& remaining_players)
   {
      return ranges::all_of(remaining_players, [&](Player player) {
         return container[as_integral(player)].has_value();
      });
   }
};

inline bool operator==(const HistorySinceBet& left, const HistorySinceBet& right)
{
   if(left.container.size() != right.container.size()) {
      return false;
   }
   return ranges::all_of(
      ranges::views::zip(left.container, right.container), [](const auto& paired_actions) {
         const auto& [left_action, right_action] = paired_actions;
         return left_action == right_action;
      });
}

class State {
  public:
   State(sptr< LeducConfig > config);

   void apply_action(Action action);
   void apply_action(Card action);
   [[nodiscard]] bool is_valid(Action action) const;
   [[nodiscard]] bool is_valid(Card outcome) const;
   [[nodiscard]] bool is_terminal() const;
   [[nodiscard]] std::vector< Action > actions() const;
   [[nodiscard]] std::vector< Card > chance_actions() const;
   [[nodiscard]] double chance_probability(Card action) const;
   [[nodiscard]] std::vector< double > payoff() const;
   [[nodiscard]] double payoff(Player player) const { return payoff()[as_integral(player)]; }

   [[nodiscard]] auto active_player() const { return m_active_player; }
   [[nodiscard]] auto card(Player player) const
   {
      return m_player_cards[as_integral(player)];
   }
   [[nodiscard]] auto public_card() const
   {
      return m_public_card;
   }
   [[nodiscard]] auto& history() const { return m_history; }
   [[nodiscard]] auto& cards() const { return m_player_cards; }

  private:
   Player m_active_player = Player::chance;
   std::vector< Player > m_remaining_players;
   std::vector< Card > m_player_cards;
   std::vector< float > m_stakes;
   std::optional< Card > m_public_card = std::nullopt;
   std::optional< Player > m_active_bettor = std::nullopt;
   float m_pot = 0.f;
   unsigned int m_bets_this_round = 0;
   HistorySinceBet m_history;
   bool m_is_terminal = false;
   bool m_terminal_checked = false;
   sptr< LeducConfig > m_config;

   static const std::vector< HistorySinceBet >& _all_terminal_histories();
   [[nodiscard]] bool _all_player_cards_assigned() const;
   [[nodiscard]] bool _has_higher_card(Player player) const;
   Player _next_active_player();
   void _single_pot_winner(std::vector< double >& payoffs, Player player) const;
   void _split_pot(std::vector< double >& payoffs, ranges::span< Player > players_with_pairs) const;
};

}  // namespace leduc

#endif  // NOR_LEDUC_POKER_STATE_HPP

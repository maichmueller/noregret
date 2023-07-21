
#ifndef NOR_LEDUC_POKER_STATE_HPP
#define NOR_LEDUC_POKER_STATE_HPP

#include <array>
#include <deque>
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

template < std::integral To = size_t, typename T >
inline To as_int(T p)
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
   double bet = 0.;
};

inline bool operator==(const Action& action1, const Action& action2)
{
   // we are never expecting large floating point values here so the absolute comparison should
   // suffice for the use case at hand
   return action1.action_type == action2.action_type
          and std::fabs(action1.bet - action2.bet) <= std::numeric_limits< double >::epsilon();
}

class LeducConfig {
  public:
   template <
      ranges::sized_range Rng1 = std::initializer_list< double >,
      ranges::sized_range Rng2 = std::initializer_list< double > >
   LeducConfig(
      size_t n_players_ = 2,
      Player starting_player_ = Player::one,
      size_t n_raises_allowed_ = 2,
      double blind_ = 1.,
      Rng1&& bet_sizes_round_one_ = {2.},
      Rng2&& bet_sizes_round_two_ = {4.},
      std::vector< Card > available_cards_ =
         {{Rank::jack, Suit::clubs},
          {Rank::jack, Suit::diamonds},
          {Rank::queen, Suit::clubs},
          {Rank::queen, Suit::diamonds},
          {Rank::king, Suit::clubs},
          {Rank::king, Suit::diamonds}}
   )
       : n_players(n_players_),
         starting_player(starting_player_),
         n_raises_allowed(n_raises_allowed_),
         blind(blind_),
         bet_sizes(ranges::to< std::vector< double > >(
            ranges::views::concat(bet_sizes_round_one_, bet_sizes_round_two_)
         )),
         bet_sizes_shapes({bet_sizes_round_one_.size(), bet_sizes_round_two_.size()}),
         available_cards(std::move(available_cards_))
   {
      if(available_cards.size() <= n_players) {
         throw std::invalid_argument(
            "There are too few cards available (" + std::to_string(available_cards.size())
            + ") for the number of players (" + std::to_string(n_players) + ").\n"
            + "At least #players + 1 (flop) many are needed."
         );
      }
      bet_sizes.shrink_to_fit();
   }

   /// returns a wider betting range confi --> increases the game tree enormously!
   /// As per Noam Brown's thesis the nr of information sets are (rest defaulted)...
   /// Leduc:      288
   /// Leduc-5:    34224
   static LeducConfig leduc5(
      size_t n_players = 2,
      Player starting_player = Player::one,
      size_t n_raises_allowed = 2,
      double blind = 1.0,
      std::vector< Card > available_cards =
         {{Rank::jack, Suit::clubs},
          {Rank::jack, Suit::diamonds},
          {Rank::queen, Suit::clubs},
          {Rank::queen, Suit::diamonds},
          {Rank::king, Suit::clubs},
          {Rank::king, Suit::diamonds}}
   )
   {
      return LeducConfig(
         n_players,
         starting_player,
         n_raises_allowed,
         blind,
         /*bet_sizes_round_one=*/{0.5, 1., 2., 4., 8.},
         /*bet_sizes_round_two=*/{1., 2., 4., 8., 16.},
         std::move(available_cards)
      );
   }

  public:
   size_t n_players;
   Player starting_player;
   size_t n_raises_allowed;
   double blind;
   std::vector< double > bet_sizes;
   std::array< size_t, 2 > bet_sizes_shapes;
   std::vector< Card > available_cards;
};

/**
 * @brief stores the currently commited action sequence
 */
class HistorySinceBet {
  public:
   HistorySinceBet(size_t n_players) : m_container(n_players, std::nullopt) {}
   HistorySinceBet(std::vector< std::optional< Action > > cont) : m_container(std::move(cont)) {}

   auto& operator[](Player player) { return m_container[as_int(player)]; }

   auto& operator[](Player player) const { return m_container[as_int(player)]; }

   auto& at(Player player) { return m_container.at(as_int(player)); }

   auto& at(Player player) const { return m_container.at(as_int(player)); }

   void reset()
   {
      ranges::for_each(m_container, [](auto& action_opt) { action_opt.reset(); });
   }

   inline bool all_acted(const std::vector< Player >& remaining_players)
   {
      return ranges::all_of(remaining_players, [&](Player player) {
         return m_container[as_int(player)].has_value();
      });
   }

   auto begin() const { return m_container.begin(); }
   auto begin() { return m_container.begin(); }
   auto end() const { return m_container.end(); }
   auto end() { return m_container.end(); }

   auto& container() { return m_container; }
   [[nodiscard]] auto& container() const { return m_container; }

  private:
   std::vector< std::optional< Action > > m_container{};
};

inline bool operator==(const HistorySinceBet& left, const HistorySinceBet& right)
{
   if(left.container().size() != right.container().size()) {
      return false;
   }
   return ranges::all_of(
      ranges::views::zip(left.container(), right.container()),
      [](const auto& paired_actions) {
         const auto& [left_action, right_action] = paired_actions;
         return left_action == right_action;
      }
   );
}

class State {
  public:
   State(LeducConfig config = {});

   template < typename... Args >
   auto apply_action(Args... args)
   {
      return apply_action({std::forward< Args >(args)...});
   }
   void apply_action(Action action);
   void apply_action(Card action);
   bool is_terminal();
   bool is_terminal() const;
   std::vector< double > payoff();
   inline double payoff(Player player) { return payoff()[as_int(player)]; }

   template < typename... Args >
   [[nodiscard]] auto is_valid(Args... args) const
   {
      return is_valid({std::forward< Args >(args)...});
   }
   [[nodiscard]] bool is_valid(Action action) const;
   [[nodiscard]] bool is_valid(Card outcome) const;
   [[nodiscard]] std::vector< Action > actions() const;
   [[nodiscard]] std::vector< Card > chance_actions() const;
   [[nodiscard]] double chance_probability(Card action) const;
   [[nodiscard]] double stake(Player player) const { return m_stakes[as_int(player)]; }

   [[nodiscard]] auto active_player() const { return m_active_player; }
   [[nodiscard]] auto remaining_players() const { return m_remaining_players; }
   [[nodiscard]] auto card(Player player) const { return m_player_cards[as_int(player)]; }
   [[nodiscard]] auto public_card() const { return m_public_card; }
   [[nodiscard]] auto& history() const { return m_history; }
   [[nodiscard]] auto& history_since_bet() const { return m_history_since_last_bet; }
   template < typename IntType >
   [[nodiscard]] auto& history_since_bet(IntType player) const
   {
      return m_history_since_last_bet[Player(player)];
   }
   [[nodiscard]] size_t round_nr() const { return m_public_card.has_value(); }
   [[nodiscard]] auto& cards() const { return m_player_cards; }
   [[nodiscard]] const auto& config() const { return m_config; }
   [[nodiscard]] auto initial_players() const
   {
      std::vector< Player > players;
      players.reserve(config().n_players);
      ranges::insert(
         players,
         players.begin(),
         ranges::views::iota(0UL, config().n_players)
            | ranges::views::transform([](int p) { return Player(p); })
      );
      return players;
   }
   [[nodiscard]] std::span< const double > bet_sizes(bool round_two) const
   {
      return std::span{config().bet_sizes}.subspan(
         size_t(round_two)
            * config().bet_sizes_shapes[0],  // start index offset is the length of
                                             // round 1's bet size vec (2nd round) or 0 (1st round)
         config().bet_sizes_shapes[round_two]  // end index is the size of round 1/2's bet size vec
      );
   }

  private:
   Player m_active_player = Player::chance;
   std::deque< Player > m_remaining_players;
   std::vector< Card > m_player_cards;
   std::vector< double > m_stakes;
   std::optional< Card > m_public_card = std::nullopt;
   std::optional< Player > m_active_bettor = std::nullopt;
   unsigned int m_bets_this_round = 0;
   HistorySinceBet m_history_since_last_bet;
   std::vector< Action > m_history{};
   bool m_is_terminal = false;
   bool m_terminal_checked = false;
   const LeducConfig m_config;

   static const std::vector< HistorySinceBet >& _all_terminal_histories();
   [[nodiscard]] bool _all_player_cards_assigned() const;
   [[nodiscard]] bool _has_higher_card(Player player) const;
   Player _cycle_active_player(bool folded, size_t shift_amount = 1);
   void _single_pot_winner(std::vector< double >& payoffs, Player player) const;

   double& _stake(Player player) { return m_stakes[as_int(player)]; }

   template < ranges::sized_range Range >
      requires std::is_same_v< ranges::value_type_t< Range >, Player >
   void _split_pot(std::vector< double >& payoffs, const Range& winners) const
   {
      double pot = std::accumulate(m_stakes.begin(), m_stakes.end(), 0.);
      double n_winners = static_cast< double >(winners.size());
      for(Player p : winners) {
         payoffs[as_int(p)] += pot / n_winners;
      };
   }
   void _determine_highest_card_winner(std::vector< Player >& winners) const;
   void _reset_order_of_play();
};

}  // namespace leduc

#endif  // NOR_LEDUC_POKER_STATE_HPP

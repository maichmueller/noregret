
#pragma once

#include <functional>
#include <range/v3/all.hpp>

#include "Action.hpp"
#include "Config.hpp"
#include "State.hpp"
#include "aze/aze.h"

namespace stratego {

class Logic {
  private:
   using Team = aze::Team;

   [[nodiscard]] virtual Logic *clone_impl() const;

  public:
   virtual ~Logic() = default;

   [[nodiscard]] uptr< Logic > clone() const { return uptr< Logic >(clone_impl()); }

   static FightOutcome handle_fight(State &state, Piece &attacker, Piece &defender);

   void apply_action(State &state, const Action &action);

   aze::Status check_terminal(const State &state);

   bool is_valid(
      const State &state,
      const Action &action,
      std::optional< Team > team_opt = std::nullopt
   );

   bool is_valid(const State &state, Move move, Team team_opt);

   template < ranges::contiguous_range Range >
   auto _valid_vectors(Position pos, Range shape, int distance = 1);

   std::vector< Action > valid_actions(const State &state, Team team);

   bool has_valid_actions(const State &state, Team team);

   void reset(State &state);

   static std::map< Position, Token > draw_setup_uniform(
      const Config &config,
      Board &curr_board,
      Team team,
      aze::utils::random::RNG &rng
   );

   static Board create_empty_board(const Config &config);

   template < std::invocable< const Config &, Board &, Team, aze::utils::random::RNG & >
                 SampleStrategyType >
   void draw_board(
      const Config &config,
      Board &curr_board,
      aze::utils::random::RNG &rng,
      SampleStrategyType setup_sampler = [](...) { return; }
   );

   void draw_board(
      const Config &config,
      Board &curr_board,
      const std::map< Team, std::map< Position, Token > > &setups
   );

   static inline FightOutcome
   fight(const Config &config, const Piece &attacker, const Piece &defender)
   {
      return fight(config, std::pair{attacker.token(), defender.token()});
   }

   static inline FightOutcome fight(const Config &config, const std::pair< Token, Token > &att_def)
   {
      return config.battle_matrix.at(att_def);
   }

   static inline void update_board(Board &board, const Position &new_pos, Piece &piece)
   {
      piece.position(new_pos);
      board[new_pos] = piece;
   }
   static inline void update_board(Board &board, const Position &new_pos)
   {
      board[new_pos] = std::nullopt;
   }

   template < std::integral IntType >
   static inline bool check_bounds(const Board &board, IntType value)
   {
      auto shape = board.shape();
      for(const auto &limit : shape) {
         if(std::cmp_less(value, 0) or std::cmp_greater_equal(value, limit)) {
            return false;
         }
      }
      return true;
   }

   /**
    * Checks boundaries of given values with respect to the board shape in every dimension.
    *
    * If the value sequence is shorter than the board dimension, then 0s are padded from the front
    * to make up for the difference.
    * @tparam Range
    * @param board
    * @param values
    * @return
    */
   template < typename Range >
      requires ranges::sized_range< Range > && ranges::bidirectional_range< Range >
               && std::integral< typename Range::value_type >
   static inline bool check_bounds(const Board &board, Range values)
   {
      auto shape = board.shape();
      if(values.size() > shape.size()) {
         return false;
      }
      return ranges::all_of(ranges::views::zip(values, shape), [&](auto v_s) {
         auto [v, s] = v_s;
         return std::cmp_greater_equal(v, 0) and std::cmp_less(v, s);
      });
   }

   template < typename Range >
      requires ranges::sized_range< Range > && ranges::bidirectional_range< Range >
               && std::integral< typename Range::value_type >
   static inline void throw_if_out_of_bounds(const Board &board, Range values)
   {
      if(not check_bounds(board, values)) {
         throw std::out_of_range(
            "Position" + std::string(values) + " out of bounds for board of shape ("
            + std::to_string(board.shape(0)) + ", " + std::to_string(board.shape(1)) + ")."
         );
      }
   }

   static void place_setup(const std::map< Position, Token > &setup, Board &board, aze::Team team);
   static void place_holes(const Config &cfg, Board &board);

   static std::map< Team, std::map< Position, Token > > extract_setup(const Board &board);
};

template < ranges::contiguous_range Range >
auto Logic::_valid_vectors(Position pos, Range shape, int distance)
{
   int right_end = static_cast< int >(shape[0]) - pos[0];
   int top_end = static_cast< int >(shape[1]) - pos[1];
   return ranges::views::concat(
             // all possible positions to the left until board ends
             ranges::views::zip(
                ranges::views::iota(std::max(-distance, -pos[0]), 0), ranges::views::repeat(0)
             ),
             // all possible positions to the right until board ends
             ranges::views::zip(
                ranges::views::iota(1, std::min(right_end, distance + 1)), ranges::views::repeat(0)
             ),
             // all possible positions to the bottom until board ends
             ranges::views::zip(
                ranges::views::repeat(0), ranges::views::iota(std::max(-distance, -pos[1]), 0)
             ),
             // all possible positions to the top until board ends
             ranges::views::zip(
                ranges::views::repeat(0), ranges::views::iota(1, std::min(top_end, distance + 1))
             )
          )
          | ranges::views::transform([](auto x_y) {
               //                  LOGD2("In _valid_vectors: ", Position(std::get< 0 >(x_y),
               //                  std::get< 1 >(x_y)));
               return Position(std::get< 0 >(x_y), std::get< 1 >(x_y));
            });
}

template <
   std::invocable< const Config &, Board &, Team, aze::utils::random::RNG & > SampleStrategyType >
void Logic::draw_board(
   const Config &config,
   Board &curr_board,
   aze::utils::random::RNG &rng,
   SampleStrategyType setup_sampler
)
{
   for(size_t i = 0; i < 2; i++) {
      auto team = Team(i);
      if(config.fixed_setups[i]) {
         place_setup(config.setups.at(team).value(), curr_board, team);
      } else {
         place_setup(setup_sampler(config, curr_board, team, rng), curr_board, team);
      }
   }
}

}  // namespace stratego

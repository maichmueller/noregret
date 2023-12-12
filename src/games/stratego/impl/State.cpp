#include "stratego/State.hpp"

#include <functional>
#include <utility>

#include "stratego/Logic.hpp"
#include "stratego/Utils.hpp"

namespace stratego {

void State::undo_last_rounds(size_t n)
{
   for(size_t i = 0; i < n; ++i) {
      auto [turn, team, move, pieces] = m_move_history.pop_last();
      m_board[move[1]] = std::move(std::get< 1 >(pieces));
      m_board[move[0]] = std::move(std::get< 0 >(pieces));
   }
   m_turn_count -= n;
}

void State::restore_to_round(size_t round)
{
   if(round > m_turn_count) [[unlikely]] {
      throw std::invalid_argument("Given round is greater than current turn count.");
   } else {
      undo_last_rounds(m_turn_count - round);
   }
}

State::State(
   Config config,
   graveyard_type graveyard,
   uptr< Logic > logic,
   Board board,
   size_t turn_count,
   const History &history,
   std::optional< std::variant< size_t, common::RNG > > seed
)
    : m_config(std::move(config)),
      m_board(std::move(board)),
      m_graveyard(std::move(graveyard)),
      m_logic(std::move(logic)),
      m_status(Status::ONGOING),
      m_status_checked(false),
      m_turn_count(turn_count),
      m_move_history(history),
      m_rng(
         seed.has_value()
            ? std::visit([](auto input) { return common::create_rng(input); }, seed.value())
            : common::create_rng()
      )
{
}

State::State(Config cfg, std::optional< std::variant< size_t, common::RNG > > seed)
    : State(
         std::move(cfg),
         graveyard_type{},
         std::make_unique< Logic >(),
         Logic::create_empty_board(cfg),
         0,
         History{},
         seed
      )
{
   Logic::place_holes(config(), board());
   std::map< Team, std::map< Position2D, Token > > setups;
   for(auto team : std::array{Team::BLUE, Team::RED}) {
      if(not config().setups.at(team).has_value()) {
         setups.emplace(team, logic()->draw_setup_uniform(config(), board(), team, rng()));
      } else {
         setups.emplace(team, config().setups.at(team).value());
      }
   }
   m_config.setups[Team::BLUE] = setups[Team::BLUE];
   m_config.setups[Team::RED] = setups[Team::RED];
   logic()->draw_board(config(), board(), setups);
   _fill_dead_pieces();
   status(Status::ONGOING);
}

void State::transition(const Action &action)
{
   status_checked() = false;
   logic()->apply_action(*this, action);
   incr_turn_count();
}

void State::transition(Move move)
{
   return transition(Action{active_team(), std::move(move)});
}

void State::_fill_dead_pieces()
{  // fill the dead pieces counter of each team if this is already an advanced configuration
   auto counters = config().token_counters;
   for(const auto &piece_opt : board()) {
      if(piece_opt.has_value()) {
         const auto &piece = piece_opt.value();
         if(piece.token() != Token::hole) {
            counters[piece.team()][piece.token()]--;
         }
      }
   }
   m_graveyard = counters;
}

std::string State::to_string() const
{
   return to_string(Team::BLUE, false);
}

std::string State::to_string(std::optional< Team > team, bool hide_unknowns) const
{
   return utils::print_board(board(), team, hide_unknowns);
}
Status State::check_terminal()
{
   return m_logic->check_terminal(*this);
}
State::State(const State &state)
    : State(
         state.config(),
         state.graveyard(),
         state.logic()->clone(),
         state.board(),
         state.turn_count(),
         state.history(),
         state.rng()
      )
{
}

}  // namespace stratego

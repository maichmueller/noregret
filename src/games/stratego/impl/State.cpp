#include "stratego/State.hpp"

#include <functional>
#include <utility>

#include "stratego/Logic.hpp"
#include "stratego/Utils.hpp"

namespace stratego {

void State::transition(const action_type &action)
{
   status_checked() = false;
   logic()->apply_action(*this, action);
   incr_turn_count();
}

void State::transition(Move move)
{
   return transition(Action{active_team(), std::move(move)});
}

State *State::clone_impl() const
{
   return new State(*this);
}
State::State(Config cfg, std::optional< std::variant< size_t, aze::utils::random::RNG > > seed)
    : base_type(Logic::create_empty_board(cfg), seed),
      m_config(std::move(cfg)),
      m_graveyard(),
      m_logic(std::make_unique< Logic >())
{
   Logic::place_holes(config(), board());
   board() = logic()->draw_board(config(), board(), rng(), &Logic::draw_setup_uniform);
   _fill_dead_pieces();
   status(Status::ONGOING);
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
   return to_string(aze::Team::BLUE, false);
}

std::string State::to_string(std::optional< Team > team, bool hide_unknowns) const
{
   return utils::print_board(board(), team, hide_unknowns);
}
aze::Status State::check_terminal()
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
       state.rng())
{
}
State &State::operator=(const State &state)
{
   base_type::operator=(state);
   m_config = state.config();
   m_graveyard = state.graveyard();
   m_logic = state.logic()->clone();
   return *this;
}

State &State::operator=(State &&) noexcept = default;
State::State(State &&) noexcept = default;
State::~State() = default;

}  // namespace stratego
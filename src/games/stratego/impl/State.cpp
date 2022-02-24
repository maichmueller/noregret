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

State *State::clone_impl() const
{
   const auto &hist = history();
   History hist_copy;
   // copy the contents of each map
   for(const size_t &turn : hist.turns()) {
      const auto &[team, action, pieces] = hist[History::Turn(turn)];
      hist_copy.commit_action(turn, team, action, pieces);
   }
   return new State(
      m_config, graveyard(), logic()->clone(), board(), turn_count(), hist_copy, rng());
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
         const auto &piece = piece_opt.m_value();
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

std::string State::to_string(aze::Team team, bool hide_unknowns) const
{
   return utils::print_board(board(), team, hide_unknowns);
}
aze::Status State::check_terminal()
{
   return m_logic->check_terminal(*this);
}

State &State::operator=(State &&) noexcept = default;
State::State(State &&) noexcept = default;
State::~State() = default;

}  // namespace stratego
#include "stratego/State.hpp"

#include <functional>
#include <utility>

#include "stratego/Logic.hpp"
#include "stratego/Utils.hpp"

namespace stratego {

void State::apply_action(const action_type &action)
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
   return new State(m_config, graveyard(), logic(), board(), turn_count(), hist_copy, rng());
}
State::State(Config cfg, std::optional< std::variant< size_t, aze::utils::random::RNG > > seed)
    : base_type(Logic::create_empty_board(cfg), seed),
      m_config(std::move(cfg)),
      m_graveyard(),
      m_logic(std::make_shared< Logic >())
{
   Logic::place_holes(config(), board());
   board() = logic()->draw_board(config(), board(), rng(), &Logic::draw_setup_uniform);
   // fill the dead pieces counter of each team
   m_graveyard = [&](const Config &config, const Board& brd) {
      auto counters = config.token_counters;
      for(const auto&piece_opt : brd) {
         if(piece_opt.has_value()) {
            const auto& piece = piece_opt.value();
            if(piece.token() != Token::hole) {
               counters[piece.team()][piece.token()]--;
            }
         }
      }
      return counters;
   }(config(), board());
   status(Status::ONGOING);
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


}  // namespace stratego
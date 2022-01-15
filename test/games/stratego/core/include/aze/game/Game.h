
#pragma once

#include <algorithm>
#include <functional>
#include <iterator>
#include <random>

#include "State.h"
#include "aze/agent/Agent.h"
#include "aze/utils/logging_macros.h"
#include "aze/utils/utils.h"

namespace aze {

template < class StateType, class LogicType, class Derived, size_t NPlayers>
class Game {
  public:
   using state_type = StateType;
   using logic_type = LogicType;
   using piece_type = typename state_type::piece_type;
   using token_type = typename piece_type::token_type;
   using position_type = typename state_type::position_type;
   using board_type = typename state_type::board_type;
   using agent_type = Agent< state_type >;

   using sptr_piece_type = sptr< piece_type >;


  private:
   static const size_t n_teams = NPlayers;
   state_type m_game_state;
   std::array< sptr< Agent< state_type > >, n_teams > m_agents;
   std::array< std::vector< sptr_piece_type >, n_teams > m_setups;

   std::vector< sptr_piece_type > extract_pieces_from_setup(
      const std::map< position_type, token_type > &setup,
      Team team);

   std::vector< sptr_piece_type > extract_pieces_from_setup(
      const std::map< position_type, sptr_piece_type > &setup,
      Team team);

  public:
   Game(
      board_type &&board,
      sptr< Agent< state_type > > ag0,
      sptr< Agent< state_type > > ag1,
      int move_count = 0);

   Game(
      const board_type &board,
      sptr< Agent< state_type > > ag0,
      sptr< Agent< state_type > > ag1,
      int move_count = 0);

   Game(const state_type &state, sptr< Agent< state_type > > ag0, sptr< Agent< state_type > > ag1);

   Game(state_type &&state, sptr< Agent< state_type > > ag0, sptr< Agent< state_type > > ag1);

   void reset(bool fixed_setups = false);

   int run_game(bool show);

   int run_step();

   void set_setup(const std::vector< sptr_piece_type > &setup, int team) { m_setups[team] = setup; }

   auto agents() { return m_agents; }
   auto agent(Team team) { return m_agents.at(team); }
   auto state() const { return m_game_state; }
   auto &state() { return m_game_state; }

   std::map< position_type, sptr_piece_type > draw_setup(Team team);
};

template < class StateType, class LogicType, class Derived, size_t n_teams >
Game< StateType, LogicType, Derived, n_teams >::Game(
   board_type &&board,
   sptr< Agent< state_type > > ag0,
   sptr< Agent< state_type > > ag1,
   int move_count)
    : m_game_state(board, move_count),
      m_agents{ag0, ag1},
      m_setups{
         extract_pieces_from_setup(m_game_state.board()->get_setup(0)),
         extract_pieces_from_setup(m_game_state.board()->get_setup(1))}
{
}

template < class StateType, class LogicType, class Derived, size_t n_teams >
Game< StateType, LogicType, Derived, n_teams >::Game(
   const board_type &board,
   sptr< Agent< state_type > > ag0,
   sptr< Agent< state_type > > ag1,
   int move_count)
    : Game(board_type(std::move(*board.clone())), ag0, ag1, move_count)
{
}

template < class StateType, class LogicType, class Derived, size_t n_teams >
Game< StateType, LogicType, Derived, n_teams >::Game(
   const state_type &state,
   sptr< Agent< state_type > > ag0,
   sptr< Agent< state_type > > ag1)
    : m_game_state(state), m_agents{ag0, ag1}
{
}

template < class StateType, class LogicType, class Derived, size_t n_teams >
Game< StateType, LogicType, Derived, n_teams >::Game(
   state_type &&state,
   sptr< Agent< state_type > > ag0,
   sptr< Agent< state_type > > ag1)
    : m_game_state(std::move(state)), m_agents{ag0, ag1}
{
}

template < class StateType, class LogicType, class Derived, size_t n_teams >
std::vector< typename Game< StateType, LogicType, Derived, n_teams >::sptr_piece_type >
Game< StateType, LogicType, Derived, n_teams >::extract_pieces_from_setup(
   const std::map< position_type, token_type > &setup,
   Team team)
{
   using val_type = typename std::map< position_type, token_type >::value_type;
   std::vector< sptr_piece_type > pc_vec;
   pc_vec.reserve(setup.size());
   std::transform(
      setup.begin(),
      setup.end(),
      std::back_inserter(pc_vec),
      [&](const val_type &pos_token) -> piece_type {
         return std::make_shared< piece_type >(pos_token.first, pos_token.second, team);
      });
   return pc_vec;
}

template < class StateType, class LogicType, class Derived, size_t n_teams >
std::vector< typename Game< StateType, LogicType, Derived, n_teams >::sptr_piece_type >
Game< StateType, LogicType, Derived, n_teams >::extract_pieces_from_setup(
   const std::map< position_type, sptr_piece_type > &setup,
   Team team)
{
   using val_type = typename std::map< position_type, sptr_piece_type >::value_type;
   std::vector< sptr_piece_type > pc_vec;
   pc_vec.reserve(setup.size());
   std::transform(
      setup.begin(),
      setup.end(),
      std::back_inserter(pc_vec),
      [&](const val_type &pos_piecesptr) -> sptr_piece_type {
         auto piece_sptr = pos_piecesptr.second;
         if(piece_sptr->team() != team)
            throw std::logic_error(
               "Pieces of team " + std::to_string(static_cast< int >(team))
               + " were expected, but received piece of team "
               + std::to_string(piece_sptr->team()));
         return piece_sptr;
      });
   return pc_vec;
}

template < class StateType, class LogicType, class Derived, size_t n_teams >
void Game< StateType, LogicType, Derived, n_teams >::reset(bool fixed_setups)
{
   auto curr_board_ptr = m_game_state.board();
   if(! fixed_setups) {
      m_setups[0] = extract_pieces_from_setup(draw_setup(0), 0);
      m_setups[1] = extract_pieces_from_setup(draw_setup(1), 1);
   }
   m_game_state = state_type(std::make_shared< board_type >(
      curr_board_ptr->get_shape(), curr_board_ptr->get_starts(), m_setups[0], m_setups[1]));
}

template < class StateType, class LogicType, class Derived, size_t n_teams >
int Game< StateType, LogicType, Derived, n_teams >::run_step()
{
   size_t turn = (m_game_state.turn_count() + 1) % 2;
   auto move = m_agents[turn]->decide_move(
      m_game_state, logic_type::get_legal_moves(*m_game_state.board(), turn));
   LOGD2("Possible Moves", logic_type::get_legal_moves(*m_game_state.board(), turn));
   LOGD2("Selected Action by team " + std::to_string(turn), move);

   int outcome = m_game_state.apply_action(move);

   return outcome;
}

template < class StateType, class LogicType, class Derived, size_t n_teams >
int Game< StateType, LogicType, Derived, n_teams >::run_game(bool show)
{
   while(true) {
      if(show)
         std::cout << m_game_state.board()->print_board(false, true);

      // test for game end status
      int outcome = m_game_state.status();

      LOGD(std::string("\n") + m_game_state.board()->print_board(false, false));
      LOGD2("Status", outcome);

      if(outcome != 404)
         return outcome;
      else
         run_step();
   }
}
}  // namespace aze
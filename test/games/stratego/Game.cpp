#include "Game.h"

namespace stratego {

Game::Game(
   const std::array< size_t, 2 > &shape,
   const std::map< position_type, int > &setup_0,
   const std::map< position_type, int > &setup_1,
   const sptr< aze::Agent< state_type > > &ag0,
   const sptr< aze::Agent< state_type > > &ag1)
    : base_type(state_type(shape, setup_0, setup_1), ag0, ag1)
{
}

Game::Game(
   const std::array< size_t, 2 > &shape,
   const std::map< position_type, token_type > &setup_0,
   const std::map< position_type, token_type > &setup_1,
   const sptr< aze::Agent< state_type > > &ag0,
   const sptr< aze::Agent< state_type > > &ag1)
    : base_type(state_type(shape, setup_0, setup_1), ag0, ag1)
{
}

Game::Game(
   size_t shape,
   const std::map< position_type, int > &setup_0,
   const std::map< position_type, int > &setup_1,
   const sptr< aze::Agent< state_type > > &ag0,
   const sptr< aze::Agent< state_type > > &ag1)
    : base_type(state_type({shape, shape}, setup_0, setup_1), ag0, ag1)
{
}

Game::Game(
   size_t shape,
   const std::map< position_type, token_type > &setup_0,
   const std::map< position_type, token_type > &setup_1,
   const sptr< aze::Agent< state_type > > &ag0,
   const sptr< aze::Agent< state_type > > &ag1)
    : base_type(state_type({shape, shape}, setup_0, setup_1), ag0, ag1)
{
}

std::map< Game::position_type, Game::sptr_piece_type > Game::draw_setup_(aze::Team team)
{
   int shape = get_state().board()->get_shape()[0];
   auto avail_types = Logic< board_type >::get_available_types(shape);

   std::vector< position_type > poss_pos = Logic< board_type >::get_start_positions(shape, team);

   std::map< position_type, sptr_piece_type > setup_out;

   std::random_device rd;
   std::mt19937 rng(rd());
   std::shuffle(poss_pos.begin(), poss_pos.end(), rng);
   std::shuffle(avail_types.begin(), avail_types.end(), rng);

   auto counter = aze::utils::counter(avail_types);

   while(! poss_pos.empty()) {
      auto &pos = poss_pos.back();
      auto &type = avail_types.back();

      setup_out[pos] = std::make_shared< piece_type >(pos, piece_type::token_type(type), team);

      poss_pos.pop_back();
      avail_types.pop_back();
   }
   return setup_out;
}

aze::Status Game::check_terminal()
{
   auto state = get_state();
   if(auto dead_pieces = state.graveyard(0);
      dead_pieces.find(token_type::flag) != dead_pieces.end()) {
      // flag of team 0 has been captured (killed), therefore team 0 lost
      return state.set_status(aze::Status::WIN_RED);
   } else if(dead_pieces = state.graveyard(1);
             dead_pieces.find(token_type::flag) != dead_pieces.end()) {
      // flag of team 1 has been captured (killed), therefore team 1 lost
      return state.set_status(aze::Status::WIN_BLUE);
   }

   // committing draw rules here

   // Rule 1: If either team has no moves left.
   if(not Logic< board_type >::has_legal_moves_(*state.board(), aze::Team::BLUE)
      or not Logic< board_type >::has_legal_moves_(*state.board(), aze::Team::RED)) {
      return state.set_status(aze::Status::TIE);
   }

   // Rule 1: The maximum turn count has been reached
   if(state.turn_count() >= state.config().max_turn_count) {
      return state.set_status(aze::Status::TIE);
   }
   return state.status();
}

}  // namespace stratego
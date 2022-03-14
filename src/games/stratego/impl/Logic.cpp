
#include "stratego/Logic.hpp"

namespace stratego {

aze::Status Logic::check_terminal(State &state)
{
   if(state.graveyard(Team::BLUE).at(Token::flag) != 0) {
      // flag of team 0 has been captured (killed), therefore team 0 lost
      return state.status(aze::Status::WIN_RED);
   } else if(state.graveyard(Team::RED).at(Token::flag) != 0) {
      // flag of team 1 has been captured (killed), therefore team 1 lost
      return state.status(aze::Status::WIN_BLUE);
   }

   // committing draw rules here

   // Rule 1: If the active team has no moves left -> loss
   if(not has_valid_actions(state, state.active_team())) {
      if(state.active_team() == Team::BLUE) {
         return state.status(Status::WIN_RED);
      } else {
         return state.status(Status::WIN_BLUE);
      }
   }

   // Rule 1: The maximum turn count has been reached
   if(std::cmp_greater_equal(state.turn_count(), state.config().max_turn_count)) {
      LOGD2("Turn count on finish: ", state.turn_count());
      return state.status(aze::Status::TIE);
   }
   return state.status(Status::ONGOING);
}

void Logic::apply_action(State &state, const Action &action)
{
   // preliminaries
   const auto &[_, from, to] = action;

   // save the access to the pieces in question
   // (removes redundant searching in board later)
   Board &board = state.board();
   auto &piece_from = board[from].value();
   auto &piece_to_opt = board[to];

   state.history().commit_action(
      state.turn_count(), state.active_team(), action, {piece_from, piece_to_opt});

   // enact the move
   if(piece_to_opt.has_value()) {
      auto &piece_to = piece_to_opt.value();
      // engage in fight, since piece_to is not a null piece
      handle_fight(state, piece_from, piece_to);
   } else {
      // no fight happened, simply move piece_from onto new position
      update_board(board, to, piece_from);
      update_board(board, from);
   }
}
FightOutcome Logic::handle_fight(State &state, Piece &attacker, Piece &defender)
{
   auto &board = state.board();
   // uncover participant pieces
   attacker.flag_hidden(false);
   defender.flag_hidden(false);
   auto outcome = fight(state.config(), attacker, defender);
   switch(outcome) {
      case FightOutcome::kill: {
         state.to_graveyard(defender);
         update_board(board, attacker.position());
         update_board(board, defender.position(), attacker);
         break;
      }
      case FightOutcome::death: {
         state.to_graveyard(attacker);
         update_board(board, attacker.position());
         break;
      }
      case FightOutcome::tie: {
         state.to_graveyard(attacker);
         state.to_graveyard(defender);
         update_board(board, attacker.position());
         update_board(board, defender.position());
         break;
      }
   }
   return outcome;
}
bool Logic::is_valid(const State &state, const Action &action, std::optional< Team > team_opt)
{
   const Team team = team_opt.value_or(action.team());
   if(action.team() != team) {
      // not this team's action
      return false;
   }
   const auto &[_, pos_before, pos_after] = action;
   const auto &board = state.board();

   if(not check_bounds(board, pos_before) or not check_bounds(board, pos_after))
      return false;

   auto p_b_opt = board[pos_before];
   auto p_a_opt = board[pos_after];

   if(not p_b_opt.has_value())
      return false;

   const auto &p_b = p_b_opt.value();

   // can't move other team's pieces
   if(p_b.team() != team) {
      return false;
   }

   // check if the target position holds a piece and whose team it belongs to
   if(p_a_opt.has_value()) {
      const auto &p_a = p_a_opt.value();
      if(p_a.team() == p_b.team())
         return false;  // cant fight
      // pieces of
      // own team
      if(p_a.token() == Token::hole) {
         return false;  // cant fight hole
      }
   }

   int move_dist = abs(pos_after[1] - pos_before[1]) + abs(pos_after[0] - pos_before[0]);

   // check if the move distance is within the move range of the token
   if(not state.config().move_ranges.at(p_b.token())(static_cast< unsigned long >(move_dist))) {
      return false;
   }

   // check for any "diagonal" moves (move not in a straight line) and, if not, for any pieces
   // blocking the way
   if(move_dist > 1) {
      if(pos_after[0] == pos_before[0]) {
         int dist = pos_after[1] - pos_before[1];
         int sign = (dist >= 0) ? 1 : -1;
         for(int i = 1; i < std::abs(dist); ++i) {
            Position pos{pos_before[0], pos_before[1] + sign * i};
            if(board[pos].has_value())
               return false;
         }
      } else if(pos_after[1] == pos_before[1]) {
         int dist = pos_after[0] - pos_before[0];
         int sign = (dist >= 0) ? 1 : -1;
         for(int i = 1; i < sign * dist; ++i) {
            Position pos = {pos_before[0] + sign * i, pos_before[1]};
            if(board[pos].has_value())
               return false;
         }
      } else
         return false;  // diagonal moves not allowed
   }
   return true;
}
std::vector< Action > Logic::valid_actions(const State &state, Team team)
{
   LOGD("Checking for valid actions.")
   const auto &board = state.board();
   std::vector< Action > actions_possible;
   for(const auto &elem : board) {
      if(elem.has_value()) {
         const auto &piece = elem.value();
         if(piece.team() == team) {
            // the position we are dealing with
            auto pos = piece.position();
            int token_move_range = 0;
            auto mr_tester = state.config().move_ranges.at(piece.token());
            for(int distance :
                ranges::views::iota(1, int(ranges::max(board.shape()))) | ranges::views::reverse) {
               if(mr_tester(static_cast< size_t >(distance))) {
                  token_move_range = distance;
                  break;
               }
            }
            ranges::for_each(
               _valid_vectors(pos, board.shape(), token_move_range), [&](const Position &pos_to) {
                  Action action{team, {pos, pos + pos_to}};
                  if(is_valid(state, action, team)) {
                     actions_possible.emplace_back(action);
                  }
               });
         }
      }
   }

   return actions_possible;
}
bool Logic::has_valid_actions(const State &state, Team team)
{
   const auto &board = state.board();
   for(const auto &piece_opt : board) {
      if(piece_opt.has_value()) {
         const auto &piece = piece_opt.value();
         if(Token token = piece.token();
            piece.team() == team && not std::set{Token::flag, Token::bomb}.contains(token)) {
            // the position we are dealing with
            auto pos = piece.position();

            //               LOGD2("check for piece", utils::enum_name(piece.token()) + " " +
            //               utils::enum_name(piece.team()));
            int token_move_range = 0;
            auto mr_tester = state.config().move_ranges.at(piece.token());
            for(int distance :
                ranges::views::iota(1, int(ranges::max(board.shape()))) | ranges::views::reverse) {
               if(mr_tester(static_cast< size_t >(distance))) {
                  token_move_range = distance;
                  break;
               }
            }

            //               auto val = ranges::any_of(
            //                  _valid_vectors(pos, board.shape(), token_move_range),
            //                  [&](const Position &value) -> bool {
            //                     return is_valid(state, Action{pos, pos + value});
            //                  });
            //               LOGD("We're here again");
            if(ranges::any_of(
                  _valid_vectors(pos, board.shape(), token_move_range),
                  [&](const Position &vector) -> bool {
                     return is_valid(state, Action{team, {pos, pos + vector}}, team);
                  })) {
               return true;
            }
         }
      }
   }
   return false;
}

std::map< Position, Token > stratego::Logic::draw_setup_uniform(
   const stratego::Config &config,
   stratego::Board &curr_board,
   stratego::Team team,
   aze::utils::random::RNG &rng)
{
   std::map< Position, Token > setup_out;

   auto start_fields = config.start_fields.at(team);
   auto token_counter = config.token_counters.at(team);
   auto uniq_token_view = token_counter | ranges::views::keys;
   std::vector< Token > tokenvec = {uniq_token_view.begin(), uniq_token_view.end()};

   ranges::shuffle(start_fields, rng);
   auto int_dist = std::uniform_int_distribution< size_t >(0, tokenvec.size() - 1);

   while(not start_fields.empty()) {
      auto &pos = start_fields.back();
      if(curr_board[pos].has_value()) {
         // if the current board already has a piece at this location, then remove the
         // position from the possible ones.
         start_fields.pop_back();
         continue;
      }
      auto choice = int_dist(rng);
      auto token = tokenvec[choice];
      // needs to be refernce since it is decremented inplace
      auto &count = token_counter[token];
      if(count > 0) {
         setup_out[pos] = token;
         count--;
         start_fields.pop_back();
      }
      if(count == 0) {
         tokenvec.erase(tokenvec.begin() + static_cast< long >(choice));
         int_dist = std::uniform_int_distribution< size_t >(0, tokenvec.size() - 1);
      }
   }
   if(not tokenvec.empty()) {
      throw std::invalid_argument(
         "Current board setup and config could not be made to agree with number of tokens to "
         "place on it.");
   }
   return setup_out;
}
Board Logic::create_empty_board(const Config &config)
{
   Board b(config.game_dims);
   for(auto x : ranges::views::iota(size_t(0), config.game_dims[0])) {
      for(auto y : ranges::views::iota(size_t(0), config.game_dims[1])) {
         b[{x, y}] = std::nullopt;
      }
   }
   return b;
}

std::map< Team, std::map< Position, Token > > Logic::extract_setup(const Board &board)
{
   std::map< Team, std::map< Position, Token > > setup;
   for(size_t i = 0; i < board.shape(0); i++) {
      for(size_t j = 0; j < board.shape(1); j++) {
         const auto &piece_opt = board[{i, j}];
         if(piece_opt.has_value()) {
            const auto &piece = piece_opt.value();
            setup[piece.team()][{int(i), int(j)}] = piece.token();
         }
      }
   }
   return setup;
}
void Logic::place_holes(const Config &cfg, Board &board)
{
   for(const auto &pos : cfg.hole_positions) {
      throw_if_out_of_bounds(board, pos);
      board[pos] = Piece(Team::NEUTRAL, pos, Token::hole);
   }
}
void Logic::place_setup(const std::map< Position, Token > &setup, Board &board, aze::Team team)
{
   for(const auto &[pos, token] : setup) {
      throw_if_out_of_bounds(board, pos);
      board[pos] = Piece(team, pos, token);
   }
}

Logic *Logic::clone_impl() const
{
   return new Logic();
}
void Logic::reset(State &state)
{
   if(not state.config().fixed_starting_team) {
      Config cfg_copy = state.config();
      cfg_copy.starting_team = aze::utils::random::choose(
         std::array{Team::BLUE, Team::RED}, state.rng());
   }
   state = State(state.config());
}
bool Logic::is_valid(const State &state, Move move, Team team)
{
   return is_valid(state, Action{team, std::move(move)});
}

}  // namespace stratego

#include "Representer.h"

namespace stratego {

/**
* We are trying to build a state representation of a Stratego board.
* To this end, 'conditions' are evaluated for each
* piece on the board. These 'conditions' are checked in sequence.
* Each condition receives its own layer representing the board with 0's
* everywhere, except for where the specific condition was true (holds value 1).
* |==========================================================================|
* |              In short: x conditions -> x binary layers                   |
* |                            (one for each condition)                      |
* |==========================================================================|
*
* Parameters
* ----------
* board: BoardType,
*   the board whose representation we want
* conditions: std::vector of tuples,
    the conditions, on which to check the board
* team: int,
    deciding which team's representation we're seeking
*
* Returns
* -------
* tensor,
    of 0's and 1's on the positions for which the relevant condition was
*   true (1) or wrong (0)
**/
Representer::condition_container Representer::_build_conditions(size_t shape)
{
   std::vector< std::tuple< token_type, int, bool > > conditions(0);
   int own_team = 0;
   auto counter = utils::counter(LogicStratego< board_type >::get_available_types(shape));
   // own m_team 0
   // [flag, 1, 2, 3, 4, ..., 10, bombs] UNHIDDEN
   for(const auto& entry : counter) {
      int type = entry.first;
      for(decltype(entry.second) version = 0; version < entry.second; ++version) {
         conditions.emplace_back(std::make_tuple(token_type(type, version), own_team, false));
      }
   }
   // [all own pieces] HIDDEN
   // Note: type and version info are unused
   // in the check in this case (thus -1)
   conditions.emplace_back(std::make_tuple(token_type(-1, -1), own_team, true));

   // enemy m_team 1
   // [flag, 1, 2, 3, 4, ..., 10, bombs] UNHIDDEN
   for(const auto& entry : counter) {
      int type = entry.first;
      for(decltype(entry.second) version = 0; version < entry.second; ++version) {
         conditions.emplace_back(std::make_tuple(token_type(type, version), 1 - own_team, false));
      }
   }
   // [all enemy pieces] HIDDEN
   // Note: type and version info are unused
   // in the check in this case (thus -1)
   conditions.emplace_back(std::make_tuple(token_type(-1, -1), 1 - own_team, true));
   return conditions;
}

std::tuple<
   std::vector< Representer::action_type >,
   std::unordered_map< Representer::token_type,
      std::vector< Representer::action_type > > >
Representer::_build_actions(size_t shape)
{
   std::vector< action_type > actions;
   std::unordered_map< token_type, std::vector< action_type > > token_to_action_map;

   const auto& available_types = LogicStratego< board_type >::get_available_types(shape);
   int curr_token = -1;
   int curr_token_version = -1;
   /*
       we want to iterate over every type of piece (as often as this type
      exists) and add the actions corresponding to its possible moves.
   */
   size_t index = 0;
   for(auto& type : available_types) {
      if(0 < type && type < 11) {
         if(curr_token != type) {
            curr_token = type;
            curr_token_version = 0;
         } else
            curr_token_version += 1;

         // if its of token 2 it can reach further -> encoded in the max_steps
         size_t max_steps = 1;
         if(type == 2)
            max_steps = shape - 1;
         // add all four directions in which the piece can walk
         std::vector< action_type > acts;
         acts.reserve(max_steps);
         std::vector< size_t > curr_indices;
         curr_indices.reserve(max_steps);
         token_type token(curr_token, curr_token_version);
         for(unsigned int i = 1; i < max_steps + 1; ++i) {
            acts.emplace_back(action_type(position_type(0, i), token, index));
            acts.emplace_back(action_type(position_type(i, 0), token, index + 1));
            acts.emplace_back(action_type(position_type(-i, 0), token, index + 2));
            acts.emplace_back(action_type(position_type(0, -i), token, index + 3));
            index = index + 4;
         }
         actions.insert(std::end(actions), std::begin(acts), std::end(acts));
         token_to_action_map[token] = acts;
      }
   }
   return std::make_tuple(actions, token_to_action_map);
}

}

#pragma once

#include <array>
#include <map>
#include <memory>
#include <numeric>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include <xtensor/xtensor.hpp>

#include "Action.h"
#include "Move.h"
#include "Position.h"
#include "aze/game/State.h"

template < typename StateType, typename DerivedType >
class RepresenterBase {
  private:
   // Convenience method for CRTP
   //
   const DerivedType *derived() const { return static_cast< const DerivedType * >(this); }
   DerivedType *derived() { return static_cast< DerivedType * >(this); }

  public:
   using state_type = StateType;
   using board_type = typename state_type::board_type;
   using position_type = typename board_type::position_type;
   using move_type = typename board_type::move_type;
   using token_type = typename state_type::token_type;

   const auto &get_actions() const { return derived()->get_actions_(); }

   // depending on the game/representation strategy of the subclass, a
   // positional variable amount of parameters can be passed to allow differing
   // implementations without knowing each use case beforehand.
   template < typename... Params >
   xt::xarray< double > state_representation(const state_type &state, const Params &...params)
   {
      return derived()->state_representation_(state, params...);
   }

   template < typename Board >
   std::vector< unsigned int > get_action_mask(const Board &board, Team team)
   {
      return derived()->get_action_mask_(board, team);
   }

   template < typename Board, typename ActionType >
   static std::vector< unsigned int >
   get_action_mask(const std::vector< ActionType > &actions, const Board &board, Team team)
   {
      return DerivedType::get_action_mask_(actions, board, team);
   }

   template < typename ValueType, size_t Dim, typename TokenType >
   inline Move< Position< ValueType, Dim > > action_to_move(
      const Position< ValueType, Dim > &pos,
      const Action< Position< ValueType, Dim >, TokenType > &action,
      Team team) const
   {
      return Move< Position< ValueType, Dim > >{pos, pos + action.get_effect()};
   }

   template < typename PieceType >
   inline typename Board< PieceType >::move_type action_to_move(
      const Board< PieceType > &board,
      const Action< typename PieceType::position_type, typename PieceType::token_type > &action,
      Team team) const
   {
      using position_type = typename PieceType::position_type;
      position_type pos = board.get_position_of_token(team, action.get_assoc_token())->second;
      return {pos, pos + action.get_effect()};
   }

   template < typename BoardType, typename HistoryType >
   inline typename BoardType::move_type action_to_move(
      const State< BoardType, HistoryType > &state,
      const Action< typename BoardType::position_type, typename BoardType::token_type > &action,
      Team team) const
   {
      return action_to_move(*state.board(), action, team);
   }

   // BSPType = Board or State or Position Type
   template < typename BSPType >
   inline typename BSPType::move_type
   action_to_move(const BSPType &bos, int action_index, Team team) const
   {
      return action_to_move(bos, get_actions()[action_index], team);
   }
};

//
// Created by Michael on 28/02/2019.
//

#pragma once

#include <functional>

#include "core/include/aze/game/Board.h"
#include "core/include/aze/game/Move.h"
#include "aze/types.h"

template < class StateType, class DerivedType >
struct Logic {
   using state_type = StateType;
   using position_type = typename state_type::position_type;
   using move_type = typename state_type::move_type;

   static bool is_legal_move(const state_type& state, const move_type& move)
   {
      return DerivedType::is_legal_move_(state, move);
   };
   static std::vector< move_type >
   get_legal_moves(const state_type& state, Team team)
   {
      return DerivedType::get_legal_moves_(state, team);
   }
   static bool has_legal_moves(const state_type& state, Team team)
   {
      return DerivedType::has_legal_moves_(state, team);
   }
};

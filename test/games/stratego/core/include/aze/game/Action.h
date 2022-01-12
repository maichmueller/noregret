
#pragma once

#include "Move.h"
#include "Position.h"

namespace aze {

template < typename Effect, typename TokenType >
struct Action {
   using effect_type = Effect;
   using token_type = TokenType;

   effect_type m_effect_vec;
   token_type m_assoc_token;
   size_t m_index;

   Action(effect_type effect, token_type piece_identifier, size_t index)
       : m_effect_vec(effect), m_assoc_token(piece_identifier), m_index(index)
   {
   }

   [[nodiscard]] effect_type get_effect() const { return m_effect_vec; }

   [[nodiscard]] token_type get_assoc_token() const { return m_assoc_token; }

   [[nodiscard]] size_t get_index() const { return m_index; }

   template < typename ValueType, size_t dim >
   Move< Position< ValueType, dim > > to_move(const Position< ValueType, dim > &pos, Team team)
   {
      return pos + m_effect_vec;
   }
};

// namespace std
}  // namespace aze
namespace std {
template < typename VectorType, typename TokenType >
struct hash< aze::Action< VectorType, TokenType > > {
   size_t operator()(const aze::Action< VectorType, TokenType > &action) const
   {
      return std::hash< TokenType >()(action.get_assoc_token());
   }
};
}  // namespace std
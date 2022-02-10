
#ifndef NOR_CONCRETE_HPP
#define NOR_CONCRETE_HPP

#include <concepts>
#include <range/v3/all.hpp>
#include <utility>
#include <vector>

#include "has.hpp"
#include "is.hpp"

namespace nor::concepts {

template < typename T >
concept action = requires(T obj)
{
   {
      obj.id()
      } -> std::same_as< bool >;
};

template < typename T, typename Action >
concept infoset = requires(T obj)
{
   T::v;
};

template < typename T, typename InSet, typename Act >
concept policy =
   action< Act > && infoset< InSet, Act > && requires(T obj, Act action, InSet info_set)
{
   std::is_move_constructible_v< T >;
   std::is_move_assignable_v< T >;
   std::is_copy_constructible_v< T >;
   std::is_copy_assignable_v< T >;

   /// getitem methods by (infoset, action) tuple and infoset only
   {
      obj[std::pair{info_set, action}]
      } -> std::convertible_to< double >;
   {
      obj[info_set]
      } -> std::convertible_to< ranges::span< Act > >;

   /// const getitem methods
} && requires(T const obj, Act action, InSet info_set)
{
   {
      obj[std::pair{info_set, action}]
      } -> std::convertible_to< double >;
   {
      obj[info_set]
      } -> std::convertible_to< ranges::span< Act > >;
};

}  // namespace nor::concepts

#endif  // NOR_CONCRETE_HPP

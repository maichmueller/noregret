
#ifndef NOR_CONCRETE_HPP
#define NOR_CONCRETE_HPP

#include <concepts>
#include <utility>
#include <vector>

#include "has.hpp"
#include "is.hpp"

namespace nor::concepts {

template < typename T >
concept Action = requires(T obj)
{
   {
      obj.id()
      } -> std::same_as< bool >;
};

template < typename T, typename Action >
concept Infoset = requires(T obj)
{
   T::v;
};

template < typename T, typename InSet, typename Act >
concept Strategy = Action< Act > && Infoset< InSet, Act > &&
   /// getitem methods by (infoset, action) tuple and infoset standalone
   requires(T obj, Act action, InSet info_set)
{
   {
      obj[std::pair{info_set, action}]
      } -> std::same_as< double >;
   {
      obj[info_set]
      } -> std::same_as< std::vector< Act > >;
   /// const getitem methods
} && requires(T const obj, Act action, InSet info_set)
{
   {
      obj[std::pair{info_set, action}]
      } -> std::same_as< double >;
   {
      obj[info_set]
      } -> std::same_as< std::vector< Act > >;
};

}  // namespace nor::concepts

#endif  // NOR_CONCRETE_HPP


#ifndef NOR_NODE_HPP
#define NOR_NODE_HPP

#include <map>
#include <vector>

#include "common/common.hpp"
#include "nor/concepts.hpp"
#include "nor/game_defs.hpp"
#include "nor/utils/utils.hpp"

namespace nor::rm {

template < typename Action, typename... OptionalData >
class InfostateNodeData {
  public:
   using regret_map_type = std::unordered_map<
      std::reference_wrapper< const Action >,
      double,
      common::ref_wrapper_hasher< const Action >,
      common::ref_wrapper_comparator< const Action > >;
   using optional_data_tuple_type = std::tuple< OptionalData... >;

   using storage_type = std::tuple< regret_map_type, OptionalData... >;

   InfostateNodeData()
      requires(sizeof...(OptionalData) == 0 or common::all_predicate_v< std::is_default_constructible, OptionalData... >)
   : m_storage(_init_storage())
   {
   }

   InfostateNodeData(OptionalData... extra_data)
      requires(sizeof...(OptionalData) > 0)
   : m_storage(_init_storage(std::move(extra_data)...))
   {
   }

   template < ranges::range ActionRange >
      requires(sizeof...(OptionalData) > 0)
   InfostateNodeData(ActionRange actions, OptionalData... optional_data)
       : m_legal_actions(), m_storage(_init_storage(std::move(optional_data)...))
   {
      emplace(std::move(actions));
   }

   template < ranges::range ActionRange >
      requires(sizeof...(OptionalData) == 0
               or common::all_predicate_v< std::is_default_constructible, OptionalData... >)
   InfostateNodeData(ActionRange actions) : m_legal_actions(), m_storage(_init_storage())
   {
      emplace(std::move(actions));
   }

   template < ranges::range ActionRange >
   void emplace(ActionRange actions)
   {
      if constexpr(concepts::is::sized< ActionRange >) {
         m_legal_actions.reserve(actions.size());
      }
      for(auto& action : actions) {
         auto& action_in_vec = m_legal_actions.emplace_back(std::move(action));
         regret().emplace(std::cref(action_in_vec), 0.);
      }
   }

   auto& actions() { return m_legal_actions; }
   auto& regret() { return std::get< regret_map_type >(m_storage); }
   auto& regret(const Action& action) { return regret()[std::ref(action)]; }

   auto& storage() { return m_storage; }
   template < size_t N = 0 >
   auto& storage_element()
   {
      return std::get< N >(m_storage);
   }
   /// if the weight is a map of some sorts, then this method will also be available for the correct
   /// key types of the map
   template < size_t N, typename T >
   auto& storage_element(const T& t)
      requires(requires(storage_type storage) { std::get< N >(storage)[t]; })
   {
      return std::get< N >(m_storage)[t];
   }

   [[nodiscard]] auto& actions() const { return m_legal_actions; }
   [[nodiscard]] auto& regret(const Action& action) const
   {
      return std::get< regret_map_type >(m_storage).at(std::ref(action));
   }
   [[nodiscard]] auto& regret() const { return std::get< regret_map_type >(m_storage); }
   [[nodiscard]] auto& storage() const { return m_storage; }
   template < size_t N = 0 >
   auto& storage_element() const
   {
      return std::get< N >(m_storage);
   }
   /// if the weight is a map of some sorts, then this method will also be available for the correct
   /// key types of the map. This method requires a std map like accessor. If there is no const
   /// bracket operator[], then we demand the function 'at' to exist. Otherwise, we cannot use this
   /// function.
   template < size_t N, typename T >
   [[nodiscard]] auto& storage_element(const T& t) const
      requires(
         sizeof...(OptionalData) > 0
         and (
            requires(const storage_type& storage) { std::get< N >(storage)[t]; }
            or requires(const storage_type& storage) { std::get< N >(storage).at(t); }
         )
      )
   {
      if constexpr(requires(const storage_type& storage) { std::get< N >(storage)[t]; }) {
         return std::get< N >(m_storage)[t];
      } else {
         return std::get< N >(m_storage).at(t);
      }
   }

  private:
   std::vector< Action > m_legal_actions;
   /// the cumulative regret the active player amassed with each action. Cumulative with regards to
   /// the number of CFR iterations. Defaults to 0 and should be updated later during the traversal.
   std::tuple< regret_map_type, OptionalData... > m_storage;

   auto _init_storage()
   {
      if constexpr(sizeof...(OptionalData) == 0) {
         return storage_type{};
      } else {
         return storage_type{
            std::tuple_cat(std::tuple{regret_map_type{}}, optional_data_tuple_type{})};
      }
   }
   auto _init_storage(regret_map_type rm)
   {
      return storage_type{std::tuple_cat(std::tuple{std::move(rm)}, optional_data_tuple_type{})};
   }
   auto _init_storage(OptionalData... data)
      requires(sizeof...(OptionalData) > 0)
   {
      return storage_type{std::tuple_cat(
         std::tuple{regret_map_type{}}, optional_data_tuple_type{std::move(data)...}
      )};
   }
   auto _init_storage(regret_map_type rm, OptionalData... data)
      requires(sizeof...(OptionalData) > 0)
   {
      return storage_type{
         std::tuple_cat(std::tuple{std::move(rm)}, optional_data_tuple_type{std::move(data)...})};
   }
};

}  // namespace nor::rm

#endif  // NOR_NODE_HPP

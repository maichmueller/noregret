

#pragma once

#include <array>
#include <iostream>
#include <vector>

namespace aze {

template < class SuperClass, class... Containers >
class IteratorCollection {
  public:
   using iterator_tuple = typename std::tuple< typename Containers::iterator... >;
   using value_tuple = typename std::tuple< typename Containers::value_type... >;

  private:
   iterator_tuple iterators;
   SuperClass *super_ptr;

   template < typename FirstIter, typename... TailIters, size_t... I >
   auto _indirection_elemwise(
      std::index_sequence< I... >,
      FirstIter &iter,
      TailIters &...tail_iters
   ) const
   {
      return std::make_tuple(
         *iter,
         std::get< I >(
            _indirection_elemwise(std::make_index_sequence< sizeof...(I) - 1 >{}, tail_iters...)
         )...
      );
   }

   template < typename LastIter >
   typename LastIter::value_type _indirection_elemwise(std::index_sequence< 0 >, LastIter &iter)
      const
   {
      return *iter;
   }

   template < typename Tuple, size_t... I >
   value_tuple _extract_iterator_values(Tuple &tuple_of_iters, std::index_sequence< I... >) const
   {
      return _indirection_elemwise(
         std::make_index_sequence< sizeof...(I) - 1 >{}, std::get< I >(tuple_of_iters)...
      );
   }

  public:
   IteratorCollection(typename Containers::iterator &&...iters)
       : iterators(std::make_tuple(iters...))
   {
   }

   value_tuple operator*() const
   {
      return _extract_iterator_values(iterators, std::index_sequence_for< Containers... >());
   }

   void set_parent_ptr(SuperClass &parent) { super_ptr = &parent; }

   SuperClass *get_parent_ptr() { return super_ptr; }

   void set_iterators(const iterator_tuple &iters) { iterators = iters; }

   iterator_tuple get_iterators() const { return iterators; }

   IteratorCollection operator++()
   {
      ++(*super_ptr);
      super_ptr->update_dep_iter(*this);
      return *this;
   }

   bool operator==(const IteratorCollection &other_it) const
   {
      auto idx_seq = std::index_sequence_for< Containers... >();
      iterator_tuple other_iterators = other_it.get_iterators();
      value_tuple other_values = _extract_iterator_values(other_iterators, idx_seq);
      value_tuple own_values = _extract_iterator_values(iterators, idx_seq);
      return own_values == other_values;
   }

   bool operator!=(const IteratorCollection &other_it) const { return ! (*this == other_it); }

   template < size_t N >
   [[nodiscard]] auto &at()
   {
      return std::get< N >(iterators);
   }

   template < size_t N >
   [[nodiscard]] const auto &at() const
   {
      return std::get< N >(iterators);
   }
};

template < class... Containers >
class Permutations {
  public:
   using value_types_tuple = typename std::tuple< typename Containers::value_type... >;
   //    using reference_tuple = typename std::tuple<typename
   //    Containers::value_type &...>; using pointer_tuple =  typename
   //    std::tuple<typename Containers::value_type *...>; using iterator_tuple
   //    = typename std::tuple<typename Containers::iterator...>;
   using iterator_tuple = IteratorCollection< Permutations, Containers... >;

   Permutations(Containers &...containers)
       : m_begin(containers.begin()...),
         m_end(containers.end()...),
         m_current(containers.begin()...)
   {
   }

   Permutations &operator++()
   {
      this->increment< sizeof...(Containers) - 1 >();
      return *this;
   }

   value_types_tuple operator*() { return *m_current; }

   iterator_tuple &begin()
   {
      m_current = m_begin;
      m_current.set_parent_ptr(*this);
      return m_current;
   }

   iterator_tuple end()
   {
      auto end = m_begin;
      end.template at< 0 >() = m_end.template at< 0 >();
      return end;
   }

   void update_dep_iter(iterator_tuple &iter) { iter.set_iterators(m_current.get_iterators()); }

   iterator_tuple get_current() { return m_current; }

  private:
   //    typedef typename extract_iterator<Containers...>::type
   //    extract_iterator_tuple;
   template < size_t N >
   typename std::enable_if_t< (N < sizeof...(Containers)) && (N > 0) > increment()
   {
      assert(m_current.template at< N >() != m_end.template at< N >());
      ++m_current.template at< N >();
      if(m_current.template at< N >() == m_end.template at< N >()) {
         m_current.template at< N >() = m_begin.template at< N >();
         this->increment< N - 1 >();
      }
   }

   template < size_t N >
   typename std::enable_if_t< N == 0 > increment()
   {
      assert(m_current.template at< N >() != m_end.template at< N >());
      ++m_current.template at< N >();
   }

   const iterator_tuple m_begin;
   const iterator_tuple m_end;
   iterator_tuple m_current;
};
}  // namespace aze

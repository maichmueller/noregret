#ifndef NOR_UTILS_HPP
#define NOR_UTILS_HPP

#include <iterator>
#include <type_traits>

namespace nor {

template <typename Iter> class ConstView {
public:
  static_assert(std::is_const_v<std::remove_reference_t<
                    typename std::iterator_traits<Iter>::reference>>,
                "ConstView can be constructed from const_iterators only.");

  ConstView(Iter begin, Iter end) : m_begin(begin), m_end(end) {}

  [[nodiscard]] auto begin() const { return m_begin; }
  [[nodiscard]] auto end() const { return m_end; }

private:
  Iter m_begin;
  Iter m_end;
};

template <typename Iter>
auto advance(Iter &&iter, typename Iter::difference_type n) {
  Iter it = std::move(iter);
  std::advance(it, n);
  return it;
}

} // namespace nor

#endif // NOR_UTILS_HPP

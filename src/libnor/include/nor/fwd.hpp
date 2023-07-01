#ifndef NOR_FWD_HPP
#define NOR_FWD_HPP

namespace nor {

template < typename T, typename... Options >
struct ActionHolder;
template < typename T, typename... Options >
struct ChanceOutcomeHolder;
template < typename T, typename... Options >
struct ObservationHolder;
template < typename T, typename... Options >
struct InfostateHolder;
template < typename T, typename... Options >
struct PublicstateHolder;
template < typename T, typename... Options >
struct WorldstateHolder;

template < typename T >
struct fosg_auto_traits;

}  // namespace nor

#endif  // NOR_FWD_HPP

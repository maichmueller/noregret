#ifndef NOR_TYPEDEFS_HPP
#define NOR_TYPEDEFS_HPP

#include <memory>

template < typename T >
using uptr = std::unique_ptr< T >;
template < typename T >
using sptr = std::shared_ptr< T >;
template < typename T >
using wptr = std::weak_ptr< T >;


#endif  // NOR_TYPEDEFS_HPP

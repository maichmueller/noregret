
#ifndef AZE_TYPES_H
#define AZE_TYPES_H

#include <memory>

template < typename T >
using uptr = std::unique_ptr< T >;
template < typename T >
using sptr = std::shared_ptr< T >;
template < typename T >
using wptr = std::weak_ptr< T >;

#endif  // AZE_TYPES_H

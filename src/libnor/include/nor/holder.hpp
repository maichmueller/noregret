#ifndef NOR_HOLDER_HPP
#define NOR_HOLDER_HPP

#include <range/v3/all.hpp>
#include <string>
#include <type_traits>
#include <vector>

#include "nor/concepts/concrete.hpp"
#include "nor/concepts/has.hpp"
#include "nor/fosg_traits.hpp"
#include "nor/game_defs.hpp"
#include "nor/tag.hpp"

namespace nor {

template <
   class DerivedHolder,
   typename Type,
   typename force_dynamic_storage = std::false_type >
   requires(
      not std::is_polymorphic_v< Type >  //
      or (
         std::is_polymorphic_v< Type >  //
         and concepts::has::method::clone_r< Type, std::unique_ptr >
         )  //
   )
struct HolderBase {
   using derived_holder_type = DerivedHolder;
   using type = std::remove_cvref_t< Type >;

   static constexpr bool force_dynamic_memory = force_dynamic_storage::value;
   static constexpr bool is_polymorphic = std::is_polymorphic_v< type >;
   static constexpr bool dynamic_storage = is_polymorphic or force_dynamic_memory;

   using value_type = std::conditional_t< dynamic_storage, sptr< type >, type >;

  protected:
   /// internal constructor to actually allocate the member
   template < typename... Ts >
   HolderBase(tag::internal_construct, Ts&&... args) noexcept(
      noexcept(this->_init_member(std::forward< Ts >(args)...))
   )
       : m_member(_init_member(std::forward< Ts >(args)...))
   {
   }

  public:
   template < typename T1, typename... Ts >
      requires common::is_none_v<
         std::remove_cvref_t< T1 >,  // remove_cvref to avoid checking each ref-case individually
         tag::internal_construct,  // don't recurse back from internal constructors or self
         HolderBase,  // don't steal the copy/move constructor calls
         derived_holder_type  // (std::remove_cvref ensures both)
         >
   explicit HolderBase(T1&& t1, Ts&&... args) noexcept(noexcept(
      HolderBase(tag::internal_construct{}, std::forward< T1 >(t1), std::forward< Ts >(args)...)
   ))
       : HolderBase(tag::internal_construct{}, std::forward< T1 >(t1), std::forward< Ts >(args)...)
   {
   }
   /// forego explicit construction from the underlying type
   template < typename T >
   HolderBase(T&& t) noexcept(noexcept(HolderBase(tag::internal_construct{}, std::forward< T >(t))))
      requires(common::is_none_v<
                 std::remove_cvref_t< T >,  // remove_cvref to avoid checking ref-case individually
                 tag::internal_construct,  // don't recurse back from internal constructors or self
                 HolderBase,  // don't steal the copy/move constructor calls
                 derived_holder_type >  // (std::remove_cvref ensures both)
              )  //
              and (
                 (std::is_base_of_v< type, std::remove_cvref_t< T > > and is_polymorphic) //
                 or std::same_as< std::remove_cvref_t< T >, type >
              )
       : HolderBase(tag::internal_construct{}, std::forward< T >(t))
   {
   }

   HolderBase() noexcept(noexcept(HolderBase(tag::internal_construct{})))
      requires std::is_default_constructible_v< type >
       : HolderBase(tag::internal_construct{})
   {
   }

   HolderBase(const HolderBase&) = default;
   HolderBase(HolderBase&&) = default;
   HolderBase& operator=(const HolderBase&) = default;
   HolderBase& operator=(HolderBase&&) = default;
   ~HolderBase() = default;
   /// implicit converion operators so that the contained type will be passable to functions that
   /// use the underlying type (similar to reference_wrapper's behaviour)
   constexpr operator const type&() const noexcept { return get(); }
   constexpr operator type&() noexcept { return get(); }

   constexpr operator const type*() const noexcept { return &get(); }
   constexpr operator type*() noexcept { return &get(); }
   /// implicit converion operators so that the passable to functions that use a shared_ptr of the
   /// underlying type
   constexpr operator const sptr< value_type >&() const noexcept
      requires dynamic_storage
   {
      return m_member;
   }
   constexpr operator sptr< value_type >&() noexcept
      requires dynamic_storage
   {
      return m_member;
   }

   /// if the holder has a polymorphic type then we can always switch specializations
   /// in the force_dynamic dimension
   template < typename any_setting >
   constexpr operator derived_holder_type&() noexcept
      requires is_polymorphic
   {
      return {m_member};
   }

   [[nodiscard]] const type& get() const noexcept
   {
      if constexpr(dynamic_storage) {
         return *m_member;
      } else {
         return m_member;
      }
   }
   [[nodiscard]] type& get() noexcept
   {
      // clang-tidy complains, but since the member function calling it is non-const, the object
      // itself is non-const. Thus, casting away the const is fine.
      return const_cast< type& >(std::as_const(*this).get());
   }

   const type& operator*() const noexcept { return get(); }
   type& operator*() noexcept { return get(); }

   [[nodiscard]] std::reference_wrapper< const type > ref() const noexcept
   {
      return std::ref(get());
   }
   [[nodiscard]] std::reference_wrapper< type > ref() { return std::ref(get()); }

   [[nodiscard]] const type* ptr() const noexcept { return &get(); }
   [[nodiscard]] type* ptr() noexcept { return &get(); }

   const type* operator->() const noexcept { return &get(); }
   type* operator->() noexcept { return &get(); }

   /// equals always compares by value of the value-type
   bool equals(const type& other) const noexcept
      requires(std::equality_comparable< type >)
   {
      return get() == other;
   }
   /// equals always compares by value of the value-type
   bool uneqals(const type& other) const noexcept
      requires(std::equality_comparable< type >)
   {
      return not equals(other);
   }
   /// compares the two objects by object identity
   /// types that are not further specififed below will never be the same as this one
   template < typename T >
   bool is(const T&) const noexcept
   {
      return false;
   }
   /// compares the two objects by object identity
   template < std::derived_from< type > T >
   bool is(const T& other) const noexcept
   {
      return ptr() == &other;
   }
   bool is(const HolderBase& other) const noexcept { return ptr() == &*other; }
   /// compares the two objects by object identity
   /// types that are not further specififed below will never be the same as this one
   template < typename T >
   bool is_not(const T&) const noexcept
   {
      return true;
   }
   /// compares the two objects by object identity
   template < std::derived_from< type > T >
   bool is_not(const T& other) const noexcept
   {
      return ptr() != &other;
   }
   bool is_not(const HolderBase& other) const noexcept { return ptr() != &*other; }

   /// equality operator always compares the value equivalence, not object equivalence
   bool operator==(const type& other) const noexcept { return equals(other); }
   bool operator!=(const type& other) const noexcept { return uneqals(other); }

   template < typename HolderTypeOut = derived_holder_type >
   [[nodiscard]] HolderTypeOut copy(
   ) const& noexcept(not dynamic_storage and noexcept(HolderTypeOut{get()}))
   {
      // internal_construct tag to avoid the implicit copy constructor
      if constexpr(is_polymorphic) {
         return HolderTypeOut{get().clone()};
      }
      return HolderTypeOut{get()};
   }

   /// rvalue ref overload for when we are calling copy on a temporary holder type
   template < typename HolderTypeOut = derived_holder_type >
   [[nodiscard]] HolderTypeOut copy() && noexcept(not dynamic_storage and noexcept(HolderTypeOut{
      get()}))
   {
      // internal_construct tag to avoid the implicit copy constructor
      constexpr auto move_pointer = [&] {
         // we are the only owner of this data, so we can safely move the pointer along
         return HolderTypeOut{std::move(m_member)};
      };
      constexpr auto clone_pointer = [&] {
         // we are not the only owner of the data, so we need to clone the data
         return HolderTypeOut{get().clone()};
      };
      constexpr auto move_data = [&] {
         // we are the only owner of the data, so we may simply move the data along
         return HolderTypeOut{std::move(get())};
      };
      constexpr auto copy_data = [&] {
         // we may not be the only owner of this memory but since it is non-polymorphic we can
         // simply copy it normally.
         return HolderTypeOut{get()};
      };

      if constexpr(is_polymorphic) {
         if(m_member.use_count() == 1) {
            return move_pointer();
         }
         return clone_pointer();
      } else if constexpr(dynamic_storage) {
         if(m_member.use_count() == 1) {
            return move_pointer();
         }
         return copy_data();
      } else {
         return move_data();
      }
   }

   [[nodiscard]] std::string to_string() const
      requires requires(type obj) {
         {
            obj.to_string()
         } -> std::same_as< std::string >;
      }
   {
      return get().to_string();
   }

  private:
   value_type m_member;

   static constexpr bool _nothrow_default_constructible = std::is_nothrow_constructible_v< type >;

   static constexpr bool _nothrow_constructible_from_lvalue =
      is_polymorphic or (not dynamic_storage and std::is_nothrow_copy_constructible_v< type >);

   value_type _init_member() const noexcept(not dynamic_storage and _nothrow_default_constructible)
      requires std::is_default_constructible_v< type >
   {
      if constexpr(dynamic_storage) {
         return std::make_shared< type >();
      } else {
         return type{};
      }
   }

   template < typename U >
   value_type _init_member(sptr< U > ptr) const
      noexcept(dynamic_storage or (not dynamic_storage and _nothrow_constructible_from_lvalue))
   {
      if constexpr(dynamic_storage) {
         // even if we are only given a pointer to the underlying type then we assume the wrapper
         // is supposed to take ownership of it and pass it to the smart pointer's constructor
         return value_type{std::move(ptr)};
      } else {
         // otherwise we simply copy the contents
         return type{*ptr};
      }
   }

   template < typename T >
   value_type _init_member(T* ptr) const noexcept(_nothrow_constructible_from_lvalue)
   {
      if constexpr(dynamic_storage) {
         // even if we are only given a pointer to the underlying type then we assume the wrapper
         // is supposed to take ownership of it and pass it to the smart pointer's constructor
         return value_type{std::move(ptr)};
      } else {
         // otherwise we simply copy the contents
         return type{*ptr};
      }
   }

   template < typename U, typename Deleter = std::default_delete< type > >
   value_type _init_member(std::unique_ptr< U, Deleter > ptr) const noexcept
   {
      if constexpr(dynamic_storage) {
         static_assert(
            std::is_convertible_v< U*, type* >,
            "Held type of unique pointer must be convertible to the contained type of the holder."
         );
         return value_type(std::move(ptr));
      } else {
         // otherwise we move the contents
         return type{std::move(*ptr.release())};
      }
   }

   template < typename T >
      requires std::is_base_of_v< type, std::remove_cvref_t< T > > and dynamic_storage
   value_type _init_member(T&& t) const
   {
      // we have to clone no matter what, since we don't know if the storage location of `t` is
      // dynamic and thus safe to take ownership of.
      if constexpr(is_polymorphic) {
         return value_type(t.clone());
      } else {
         return value_type(std::make_shared< type >(std::forward< T >(t)));
      }
   }

   template < typename T >
      requires std::is_base_of_v< type, std::remove_cvref_t< T > > and (not dynamic_storage)
   value_type _init_member(T&& t) const noexcept(noexcept(type{std::forward< T >(t)}))
   {
      return type{std::forward< T >(t)};
   }

   template < typename FirstArg, typename... Args >
      requires(  //
         not std::is_base_of_v< type, std::remove_cvref_t< FirstArg > >  //
         and not std::same_as< std::remove_cvref_t< FirstArg >, tag::internal_construct >  //
      )
   [[nodiscard]] value_type _init_member(FirstArg&& first_arg, Args&&... args) const
      noexcept(noexcept(
         this->_init_member_impl(std::forward< FirstArg >(first_arg), std::forward< Args >(args)...)
      ))
   {
      return this->_init_member_impl(
         std::forward< FirstArg >(first_arg), std::forward< Args >(args)...
      );
   }

   template < typename... Ts >
      requires(sizeof...(Ts) > 0) and (not dynamic_storage)
              and concepts::brace_initializable< type, Ts... >
   value_type _init_member_impl(Ts&&... ts) const
      noexcept(noexcept(type{std::forward< Ts >(ts)...}))
   {
      return type{std::forward< Ts >(ts)...};
   }

   template < typename... Ts >
      requires(sizeof...(Ts) > 0) and dynamic_storage
              and concepts::brace_initializable< type, Ts... >
   value_type _init_member_impl(Ts&&... ts) const
   {
      return std::make_shared< type >(std::forward< Ts >(ts)...);
   }
};

template < typename Action, typename... Options >
struct BasicHolder: public HolderBase< ActionHolder< Action, Options... >, Action, Options... > {
   using base = HolderBase< ActionHolder< Action, Options... >, Action, Options... >;
   using base::base;
};

template < typename Action, typename... Options >
struct ActionHolder: public HolderBase< ActionHolder< Action, Options... >, Action, Options... > {
   using base = HolderBase< ActionHolder< Action, Options... >, Action, Options... >;
   using base::base;
};

template < typename ChOutcome, typename... Options >
struct ChanceOutcomeHolder:
    public HolderBase< ChanceOutcomeHolder< ChOutcome, Options... >, ChOutcome, Options... > {
   using base = HolderBase< ChanceOutcomeHolder< ChOutcome, Options... >, ChOutcome, Options... >;
   using base::base;
};

template < typename Observation, typename... Options >
struct ObservationHolder:
    public HolderBase< ObservationHolder< Observation, Options... >, Observation, Options... > {
   using base = HolderBase< ObservationHolder< Observation, Options... >, Observation, Options... >;
   using base::base;
};

template < typename Worldstate, typename... Options >
struct WorldstateHolder:
    public HolderBase< WorldstateHolder< Worldstate, Options... >, Worldstate, Options... > {
   using base = HolderBase< WorldstateHolder< Worldstate, Options... >, Worldstate, Options... >;
   using base::base;
};

template < typename Infostate, typename... Options >
struct InfostateHolder:
    public HolderBase< InfostateHolder< Infostate, Options... >, Infostate, Options... > {
   using base = HolderBase< InfostateHolder< Infostate, Options... >, Infostate, Options... >;
   using base::base;
   using base::get;
   using observation_type = auto_observation_type< Infostate >;

   [[nodiscard]] size_t size() const { return get().size(); }

   void update(
      const ObservationHolder< observation_type >& public_obs,
      const ObservationHolder< observation_type >& private_obs
   )
   {
      return get().update(public_obs, private_obs);
   }

   auto& operator[](auto index) { return get()[size_t(index)]; }

   const auto& operator[](auto index) const { return get()[size_t(index)]; }

   [[nodiscard]] Player player() const { return get().player(); }
};

template < typename Publicstate, typename... Options >
struct PublicstateHolder:
    public HolderBase< PublicstateHolder< Publicstate, Options... >, Publicstate, Options... > {
   using base = HolderBase< PublicstateHolder< Publicstate, Options... >, Publicstate, Options... >;
   using base::base;
   using base::get;
   using observation_type = auto_observation_type< Publicstate >;

   [[nodiscard]] size_t size() const { return get().size(); }

   const auto& update(const observation_type& obs) { return get().update(obs); }

   auto& operator[](auto index) { return get()[size_t(index)]; }

   const auto& operator[](auto index) const { return get()[size_t(index)]; }
};

}  // namespace nor
namespace std {

template < typename T >
   requires common::logical_or_v<
      common::is_specialization< std::remove_cvref_t< T >, nor::ActionHolder >,
      common::is_specialization< std::remove_cvref_t< T >, nor::ChanceOutcomeHolder >,
      common::is_specialization< std::remove_cvref_t< T >, nor::ObservationHolder >,
      common::is_specialization< std::remove_cvref_t< T >, nor::InfostateHolder >,
      common::is_specialization< std::remove_cvref_t< T >, nor::PublicstateHolder > >
struct hash< T > {
   size_t operator()(const T& t) const { return std::hash< typename T::type >{}(t); }
};

}  // namespace std

namespace common {

// extend this template to the holder cases
template < typename T >
// clang-format off
   requires(any_of<
      is_specialization_v< ::std::remove_cvref_t< T >, ::nor::ActionHolder >,
      is_specialization_v< ::std::remove_cvref_t< T >, ::nor::ChanceOutcomeHolder >,
      is_specialization_v< ::std::remove_cvref_t< T >, ::nor::ObservationHolder >,
      is_specialization_v< ::std::remove_cvref_t< T >, ::nor::InfostateHolder >,
      is_specialization_v< ::std::remove_cvref_t< T >, ::nor::PublicstateHolder >,
      is_specialization_v< ::std::remove_cvref_t< T >, ::nor::WorldstateHolder >
      >
   )
decltype(auto)  // clang-format on
deref(T&& t)
{
   return *std::forward< T >(t);
}

// these specializations are not needed (and in fact lead to compiler errors due to ambiguity with
// the specializations of the underlying type and implicit conversions to said type)

// template < typename T >
//    requires(any_of<
//             is_specialization_v< T, ::nor::ActionHolder >,
//             is_specialization_v< T, ::nor::ChanceOutcomeHolder >,
//             is_specialization_v< T, ::nor::ObservationHolder >,
//             is_specialization_v< T, ::nor::InfostateHolder >,
//             is_specialization_v< T, ::nor::PublicstateHolder >,
//             is_specialization_v< T, ::nor::WorldstateHolder > >)
// inline std::string to_string(const T& value)
//{
//    return to_string(value.get());
// }

//// holders are printable if the contained type is printable
// template < typename T >
//    requires(any_of<
//             is_specialization_v< T, ::nor::ActionHolder >,
//             is_specialization_v< T, ::nor::ChanceOutcomeHolder >,
//             is_specialization_v< T, ::nor::ObservationHolder >,
//             is_specialization_v< T, ::nor::InfostateHolder >,
//             is_specialization_v< T, ::nor::PublicstateHolder >,
//             is_specialization_v< T, ::nor::WorldstateHolder > >)
// struct printable< T >: public printable< typename T::type > {};

}  // namespace common

#endif  // NOR_HOLDER_HPP

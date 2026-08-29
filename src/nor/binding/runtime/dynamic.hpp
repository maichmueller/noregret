#ifndef NOR_BINDING_RUNTIME_DYNAMIC_HPP
#define NOR_BINDING_RUNTIME_DYNAMIC_HPP

#include <memory>

#include "types.hpp"

namespace nor::binding::runtime {

/// Separate extension point for a later Nanobind trampoline.  A dynamic game is intentionally not
/// convertible to GameHandle and has no entry in the immutable static catalog.  Implementations
/// may use virtual callbacks internally; the static path above never sees them.
class DynamicSolverSession {
  public:
   virtual ~DynamicSolverSession() = default;

   virtual Result< IterateResult > iterate(size_t) = 0;
   virtual Result< IterateResult > advance(size_t) = 0;
   virtual Result< TraceResult > trace(const TraceRequest&) const = 0;
   virtual Result< SessionStats > stats() const = 0;
   virtual Result< PolicyView > policy_view(PolicyViewKind) const = 0;
};

class DynamicGameBoundary {
  public:
   virtual ~DynamicGameBoundary() = default;

   [[nodiscard]] virtual GameSpec game_spec() const = 0;
   [[nodiscard]] virtual Result< std::unique_ptr< DynamicSolverSession > >
      make_session(SolverId, ProfileId, SessionOptions) const = 0;
};

/// A dynamic handle is explicit at every call site.  There is deliberately no overload of
/// make_session accepting this type, which prevents a Python-authored game from silently entering
/// the compile-time registry.
class DynamicGameHandle {
  public:
   explicit DynamicGameHandle(std::shared_ptr< const DynamicGameBoundary > boundary)
       : m_boundary(std::move(boundary))
   {
   }

   [[nodiscard]] explicit operator bool() const noexcept { return bool(m_boundary); }

   [[nodiscard]] const DynamicGameBoundary* boundary() const noexcept { return m_boundary.get(); }

   [[nodiscard]] Result< std::unique_ptr< DynamicSolverSession > >
   make_session(SolverId solver, ProfileId profile, SessionOptions options = {}) const
   {
      if(not m_boundary) {
         return std::unexpected(CapabilityError{
            .code = CapabilityErrorCode::invalid_handle,
            .message = "dynamic game handle is empty",
            .solver = solver,
            .profile = profile});
      }
      return m_boundary->make_session(solver, profile, options);
   }

   [[nodiscard]] Result< GameSpec > game_spec() const
   {
      if(not m_boundary) {
         return std::unexpected(CapabilityError{
            .code = CapabilityErrorCode::invalid_handle, .message = "dynamic game handle is empty"}
         );
      }
      return m_boundary->game_spec();
   }

  private:
   std::shared_ptr< const DynamicGameBoundary > m_boundary;
};

}  // namespace nor::binding::runtime

#endif  // NOR_BINDING_RUNTIME_DYNAMIC_HPP


#pragma once

#include <aze/aze.h>

#include "Piece.h"
#include "array"
#include "map"
#include "memory"
#include "vector"

namespace stratego {

class Board: public aze::Board< Piece > {
  public:
   using base_type = Board< PieceStratego >;

   // also specializing one
   Board(
      const std::array< size_t, 2 > &shape,
      const std::map< position_type, int > &setup_0,
      const std::map< position_type, int > &setup_1)
       : base_type(shape, adapt_setup(setup_0), adapt_setup(setup_1))
   {
      _add_obstacles();
   }

   // decorating the constructors with add_obstacles after construction
   template < typename... Params >
   Board(Params... params) : base_type(params...)
   {
      _add_obstacles();
   }

   [[nodiscard]] std::string print_board(aze::Team team, bool hide_unknowns) const override;

  private:
   void _add_obstacles();
   static std::vector< sptr< piece_type > > adapt_setup(
      const std::map< position_type, int > &setup);

   [[nodiscard]] Board *clone_impl() const override;
};

}  // namespace stratego

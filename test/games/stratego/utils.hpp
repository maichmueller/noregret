#pragma once

#include <aze/aze.h>
#include "StrategoDefs.hpp"
#include <string>
#include <xtensor/xtensor.hpp>

namespace stratego::utils {

template < typename DType>
std::string print_board(xt::xtensor<DType, 2> board, aze::Team team, bool hide_unknowns)
{
   using Team = aze::Team;

#define VERT_BAR "\u2588"
#define RESET "\x1B[0m"
#define BLUE "\x1B[44m"
#define RED "\x1B[41m"

   int H_SIZE_PER_PIECE = 9;
   int V_SIZE_PER_PIECE = 3;
   // the space needed to assign row indices to the rows and to add a splitting
   // bar "|"
   int row_ind_space = 4;
   int dim_x = board.shape(0);
   int dim_y = board.shape(1);
   int mid = V_SIZE_PER_PIECE / 2;

   // piece string lambda function that returns a str of the kin
   // "-1 \n
   // 10.1 \n
   //   1"
   auto create_piece_str = [&H_SIZE_PER_PIECE, &mid, &team, &hide_unknowns](
                              const auto &piece, int line) {
      if(piece.is_null())
         return std::string(static_cast< unsigned long >(H_SIZE_PER_PIECE), ' ');

      std::string color = BLUE;  // blue by default (for team 0)
      if(piece.get_team() == -1 && ! piece.is_null())
         // piece is an obstacle (return a gray colored field)
         return "\x1B[30;47m" + aze::utils::center("", H_SIZE_PER_PIECE, " ") + RESET;
      else if(piece.get_team(static_cast< bool >(team)) == 1) {
         color = RED;  // background red, text "white"
      }
      if(line == mid - 1) {
         // hidden info line
         std::string h = piece.get_flag_hidden() ? "?" : " ";
         return color + aze::utils::center(h, H_SIZE_PER_PIECE, " ") + RESET;
      } else if(line == mid) {
         // type and version info line
         if(hide_unknowns && piece.get_flag_hidden() && piece.get_team(static_cast< bool >(team))) {
            return color + std::string(static_cast< unsigned long >(H_SIZE_PER_PIECE), ' ') + RESET;
         }
         const auto &token = piece.get_token();
         return color
                + aze::utils::center(
                   std::to_string(static_cast< int >(token)), H_SIZE_PER_PIECE, " ")
                + RESET;
      } else if(line == mid + 1)
         // team info line
         // return color + center(std::to_string(piece.get_team(flip_board)),
         // H_SIZE_PER_PIECE, " ") + reset;
         return color + aze::utils::center("", H_SIZE_PER_PIECE, " ") + RESET;
      else
         // empty line
         return std::string(static_cast< unsigned long >(H_SIZE_PER_PIECE), ' ');
   };

   std::stringstream board_print;
   board_print << "\n";

   std::string init_space = std::string(static_cast< unsigned long >(row_ind_space), ' ');
   std::string h_border = aze::utils::repeat(
      VERT_BAR, static_cast< unsigned long >(dim_x * (H_SIZE_PER_PIECE + 1) - 1));

   board_print << init_space << VERT_BAR << h_border << VERT_BAR << "\n";
   std::string init = board_print.str();
   sptr< DType > curr_piece;

   // row means row of the board. not actual rows of console output.
   for(int row = dim_y - 1; row > - 1;
       --row) {  // iterate backwards through the rows for correct display
      // per piece we have V_SIZE_PER_PIECE many lines to fill consecutively.
      // Iterate over every column and append the new segment to the right line.
      std::vector< std::stringstream > line_streams(static_cast< unsigned int >(V_SIZE_PER_PIECE));

      for(int col = 0; col < dim_x; ++col) {
         if(static_cast< bool >(team)) {
            curr_piece = board(dim_x - 1 - row, dim_y - 1 - col);
         } else
            curr_piece = board(row, col);

         for(int i = 0; i < V_SIZE_PER_PIECE; ++i) {
            std::stringstream curr_stream;

            if(i == mid - 1 || i == mid + 1) {
               if(col == 0) {
                  curr_stream << std::string(static_cast< unsigned long >(row_ind_space), ' ');
               }
               curr_stream << VERT_BAR << create_piece_str(*curr_piece, i);
            } else if(i == mid) {
               if(col == 0) {
                  if(row < 10)
                     curr_stream << " " << row;
                  else
                     curr_stream << row;

                  curr_stream << std::string(static_cast< unsigned long >(row_ind_space - 2), ' ')
                              << VERT_BAR;
               }
               curr_stream << create_piece_str(*curr_piece, i);
               if(col != dim_x - 1)
                  curr_stream << VERT_BAR;
            }
            // extend the current line i by the new information
            line_streams[i] << curr_stream.str();
         }
      }
      for(auto &stream : line_streams) {
         board_print << stream.str() << VERT_BAR << "\n";
      }
      board_print << init_space << VERT_BAR << h_border << VERT_BAR << "\n";
   }
   // column width for the row index plus vertical dash
   board_print << std::string(static_cast< unsigned long >(row_ind_space), ' ');
   // print the column index rows
   for(int i = 0; i < dim_x; ++i) {
      board_print << aze::utils::center(std::to_string(i), H_SIZE_PER_PIECE + 1, " ");
   }
   board_print << "\n";
   return board_print.str();
}
}  // namespace stratego::utils
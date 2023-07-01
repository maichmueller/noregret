
#ifndef NOR_TAG_HPP
#define NOR_TAG_HPP

namespace nor::tag {

struct normalize {};

struct current_policy {};
struct average_policy {};

struct internal_construct {};

struct action {};
struct chance_outcome {};
struct observation {};
struct infostate {};
struct publicstate {};
struct worldstate {};

struct inplace {};

}  // namespace nor::tag

#endif  // NOR_TAG_HPP

#pragma once

#ifndef NDEBUG
   #define LOGE2(x, y)                                                                           \
                                                                                                 \
      std::cout << "ERROR: "                                                                     \
                << "[" << __FILE__ << "][" << __FUNCTION__ << "][Line " << __LINE__ << "] " << x \
                << ": " << y << std::endl  // NOLINT
   #define LOGI2(x, y)                                                                           \
      std::cout << "INFO : "                                                                     \
                << "[" << __FILE__ << "][" << __FUNCTION__ << "][Line " << __LINE__ << "] " << x \
                << ": " << y << std::endl  // NOLINT
   #define LOGD2(x, y)                                                                           \
      std::cout << "DEBUG: "                                                                     \
                << "[" << __FILE__ << "][" << __FUNCTION__ << "][Line " << __LINE__ << "] " << x \
                << ": " << y << std::endl  // NOLINT
   #define LOGE(x)                                                                               \
      std::cout << "ERROR: "                                                                     \
                << "[" << __FILE__ << "][" << __FUNCTION__ << "][Line " << __LINE__ << "] " << x \
                << std::endl  // NOLINT
   #define LOGI(x)                                                                               \
      std::cout << "INFO : "                                                                     \
                << "[" << __FILE__ << "][" << __FUNCTION__ << "][Line " << __LINE__ << "] " << x \
                << std::endl  // NOLINT
   #define LOGD(x)                                                                               \
      std::cout << "DEBUG: "                                                                     \
                << "[" << __FILE__ << "][" << __FUNCTION__ << "][Line " << __LINE__ << "] " << x \
                << std::endl  // NOLINT
#else

   #define LOGE2(x, y) NULL
   #define LOGI2(x, y) NULL
   #define LOGD2(x, y) NULL
   #define LOGE(x) NULL
   #define LOGI(x) NULL
   #define LOGD(x) NULL

#endif
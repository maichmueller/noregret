# use (i.e. don't skip) the full RPATH for the build tree
set(CMAKE_SKIP_BUILD_RPATH FALSE)

# when building, don't use the install RPATH already (but later on when installing)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")

# add the automatically determined parts of the RPATH which point to directories outside the build tree to the install
# RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# the RPATH to be used when installing, but only if it's not a system directory
list(
    FIND
    CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES
    "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
    isSystemDir)
if("${isSystemDir}" STREQUAL "-1")
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
endif("${isSystemDir}" STREQUAL "-1")

if(INSTALL_PYMODULE)
    message("Configuring installation for python module.")

    # A wheel is relocatable, so the extension has to find its siblings relative to itself rather than through an
    # absolute prefix that will not exist on the installing machine.
    if(APPLE)
        set(_pymodule_origin "@loader_path")
    else()
        set(_pymodule_origin "$ORIGIN")
    endif()
    set_target_properties(${nor_pymodule} PROPERTIES INSTALL_RPATH "${_pymodule_origin}")

    # The extension links the game libraries that are built as real libraries. Installing only the extension produces a
    # wheel that imports and then immediately fails on a missing .so, so they travel with it. Header-only games
    # contribute no runtime file and are skipped.
    set(_pymodule_runtime_libraries)
    foreach(_pymodule_game IN LISTS NOR_GAME_LIBRARIES)
        if(TARGET ${_pymodule_game})
            get_target_property(_pymodule_game_type ${_pymodule_game} TYPE)
            if(_pymodule_game_type STREQUAL "SHARED_LIBRARY" OR _pymodule_game_type STREQUAL "MODULE_LIBRARY")
                set_target_properties(${_pymodule_game} PROPERTIES INSTALL_RPATH "${_pymodule_origin}")
                list(APPEND _pymodule_runtime_libraries ${_pymodule_game})
            endif()
        endif()
    endforeach()

    install(TARGETS ${nor_pymodule} ${_pymodule_runtime_libraries} LIBRARY DESTINATION nor)

    unset(_pymodule_origin)
    unset(_pymodule_game)
    unset(_pymodule_game_type)
    unset(_pymodule_runtime_libraries)
else()
    message("Configuring Installation For C++ Library.")
    #
    # Install pkg-config file
    #

    set(NOR_PKGCONFIG ${CMAKE_CURRENT_BINARY_DIR}/nor.pc)

    configure_file(${_cmake_DIR}/in/nor.pc.in ${NOR_PKGCONFIG} @ONLY)

    install(FILES ${NOR_PKGCONFIG} DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)

    write_basic_package_version_file(
        "${PROJECT_BINARY_DIR}/${PROJECT_NAME_LOWERCASE}ConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY AnyNewerVersion)

    configure_package_config_file(
        "${_cmake_DIR}/in/${PROJECT_NAME_LOWERCASE}Config.cmake.in"
        "${PROJECT_BINARY_DIR}/${PROJECT_NAME_LOWERCASE}Config.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME_LOWERCASE}/cmake/)

    # CMake requires all interface dependencies of our library targets, including options and warnings, to be exported.
    install(TARGETS project_options EXPORT ${PROJECT_NAME_LOWERCASE}Options)
    install(TARGETS project_warnings EXPORT ${PROJECT_NAME_LOWERCASE}Warnings)
    install(
        TARGETS ${nor_lib} common required_min_libs
        EXPORT ${PROJECT_NAME_LOWERCASE}Targets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT Runtime
        BUNDLE DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT Runtime
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Runtime
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Development
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT Development)

    if(ENABLE_BUILD_PYTHON_EXTENSION)
        # to use the latest build of the library for a development package install (pip install path/to/project -e), we
        # configure an optional install component of the python extension in-source
        install(
            TARGETS ${nor_pymodule}
            LIBRARY DESTINATION ${PROJECT_PYNOR_DIR}
                    COMPONENT PyExtension_inplace
                    OPTIONAL)
        # this target can be called conveniently from within IDEs to replace the current build of a development install
        add_custom_target(
            install_${nor_pymodule}_insource
            COMMAND ${CMAKE_COMMAND} --install . --component PyExtension_inplace
            DEPENDS ${nor_pymodule})
    endif()

    install(
        EXPORT ${PROJECT_NAME_LOWERCASE}Options
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME_LOWERCASE}/cmake/
        NAMESPACE ${PROJECT_NAME_LOWERCASE}::)
    install(
        EXPORT ${PROJECT_NAME_LOWERCASE}Warnings
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME_LOWERCASE}/cmake/
        NAMESPACE ${PROJECT_NAME_LOWERCASE}::)
    install(
        EXPORT ${PROJECT_NAME_LOWERCASE}Targets
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME_LOWERCASE}/cmake/
        NAMESPACE ${PROJECT_NAME_LOWERCASE}::)

    install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME_LOWERCASE}ConfigVersion.cmake"
                  "${PROJECT_BINARY_DIR}/${PROJECT_NAME_LOWERCASE}Config.cmake"
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME_LOWERCASE}/cmake/)
    # this installation assumes that the project has an eponymous include directory for the project c++ library we
    # install that sub-directory instead of PROJECT_nor_INCLUDE_DIR, in order to avoid an include/include/proj_name
    # situation and get the correct include/proj_name.
    install(DIRECTORY ${PROJECT_NOR_INCLUDE_DIR}/${PROJECT_NAME_LOWERCASE} DESTINATION include)

    export(PACKAGE nor)

endif()

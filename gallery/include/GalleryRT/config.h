#pragma once
/**
  * @file config.h
  * @brief GalleryRT configuration header for DLL export/import definitions.
  * @author sailing-innocent
  * @date 2025-08-26
  */
#ifdef _MSC_VER
#ifdef GALLERY_DLL_EXPORTS
    #define GALLERY_API __declspec(dllexport)
#else
    #define GALLERY_API __declspec(dllimport)
#endif
#else
    #define GALLERY_API __attribute__((visibility("default")))
#endif


#ifdef ANDROID_PLATFORM
    #include <android/log.h>
    #define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "Debug", ##__VA_ARGS__);
#endif


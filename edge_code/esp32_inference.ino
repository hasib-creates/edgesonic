#include <Arduino.h>
#include <cmath>
#include <ESP_I2S.h>          
#include <esp_dsp.h>

#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include "model_data.h"
#include "mel_filterbank.h"
#include "svdd_center.h"

#define VERBOSE 1

constexpr int   SAMPLE_RATE = 16000;
constexpr int   N_FFT       = 512;
constexpr int   HOP_LEN     = 256;
constexpr int   MEL_BINS    = 16;
constexpr int   FRAMES      = 64;

constexpr float MEAN = -5.0f;
constexpr float STD  =  4.5f;
constexpr float SVDD_THR = 1.5f;
constexpr float MSE_THR  = 0.2f;

constexpr int PDM_CLK  = 42;
constexpr int PDM_DIN  = 41;

constexpr int WIN_SAMPLES = (FRAMES - 1) * HOP_LEN + N_FFT;
constexpr int CHUNK       = 1024;

static I2SClass I2S;
static int16_t *pcm  = nullptr;
static float   *fft  = nullptr;
static float   *hann = nullptr;
static float   *twid = nullptr;
static size_t   pcm_idx = 0;

namespace {
constexpr size_t ARENA_BYTES = 80 * 1024;
uint8_t* arena  = nullptr;
const tflite::Model* model  = nullptr;
tflite::MicroInterpreter* interp = nullptr;
TfLiteTensor *tinp = nullptr, *latent_out = nullptr, *recon_out = nullptr;
float in_scale = 1.f;  int32_t in_zp  = 0;
float latent_out_scale = 1.f; int32_t latent_out_zp = 0;
float recon_out_scale = 1.f; int32_t recon_out_zp = 0;
}

#include <WiFi.h>
#include <PubSubClient.h>
#include <WireGuard-ESP32.h>

/* WiFi credentials - Using your working setup */
const char* WIFI_SSID = "HASIB-PC 3007";
const char* WIFI_PASS = "98O1+j19";

/* WireGuard settings - Using your working setup */
const char* WG_PRIVATE_KEY = "mJFnTuiV1qk910oyVrtHzNDxJN+h2uKy1zt84TEryng=";
const char* WG_PEER_PUBLIC_KEY = "zPlr7iSEa5V6yt34p/kTqTznpa5nzMgUcwFKatmKvFk=";

const char* WG_ENDPOINT = "192.168.32.211"; // PC's WiFi IP

uint16_t    WG_PORT = 51820;
IPAddress WG_LOCAL_IP(10, 8, 0, 2);
// IPAddress WG_LOCAL_IP(10, 8, 0, 3);

/* MQTT settings - Using tunnel IP from your working setup */
const char* MQTT_BROKER = "10.8.0.1"; // PC's WireGuard tunnel IP
const int   MQTT_PORT = 1883;
const char* MQTT_TOPIC = "sensors/esp32_1/data"; // Using your working topic

static WireGuard wg;
static WiFiClient wifiClient;
static PubSubClient mqtt(wifiClient);
static bool mqtt_enabled = false;  // Flag to disable MQTT if it's causing issues

/* MQTT callback for incoming messages */
static void mqttCallback(char* topic, byte* payload, unsigned int length) {
  // Handle incoming messages if needed
  if (topic && payload && length > 0) {
    Serial.printf("[MQTT] Message arrived [%s]: ", topic);
    for (unsigned int i = 0; i < length; i++) {
      Serial.print((char)payload[i]);
    }
    Serial.println();
  }
}

/* MQTT connection helper */
static bool mqttReconnect() {
  if (!mqtt_enabled) return false;
  
  if (mqtt.connected()) return true;
  
  Serial.print("[MQTT] Attempting connection...");
  
  // Add a small delay to prevent rapid reconnection attempts
  delay(500);
  
  bool connected = false;
  try {
    connected = mqtt.connect("ESP32S3_ML_Client1");
  } catch (...) {
    Serial.println(" EXCEPTION during connect!");
    mqtt_enabled = false;  // Disable MQTT to prevent further crashes
    return false;
  }
  
  if (connected) {
    Serial.println(" connected");
    return true;
  } else {
    int state = mqtt.state();
    Serial.printf(" failed, rc=%d", state);
    
    // Decode MQTT error codes for better debugging
    switch(state) {
      case -4: Serial.println(" (connection timeout)"); break;
      case -3: Serial.println(" (connection lost)"); break;
      case -2: Serial.println(" (connect failed)"); break;
      case -1: Serial.println(" (disconnected)"); break;
      case  1: Serial.println(" (bad protocol)"); break;
      case  2: Serial.println(" (bad client ID)"); break;
      case  3: Serial.println(" (unavailable)"); break;
      case  4: Serial.println(" (bad credentials)"); break;
      case  5: Serial.println(" (unauthorized)"); break;
      default: Serial.println(" (unknown error)"); break;
    }
    
    // If we get multiple failures, disable MQTT
    static int failure_count = 0;
    failure_count++;
    if (failure_count > 3) {
      Serial.println("[MQTT] Too many failures, disabling MQTT");
      mqtt_enabled = false;
    }
    
    return false;
  }
}

/* Publish helper - Updated to use your working MQTT topic and format */
static void publishScores(float svdd, float mse)
{
  if (!mqtt_enabled) {
    // Just print to serial if MQTT is disabled
    Serial.printf("[DATA] SVDD=%.3f MSE=%.4f (MQTT disabled)\n", svdd, mse);
    return;
  }
  
  // Check WiFi connection first
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[MQTT] WiFi disconnected, skipping publish");
    return;
  }
  
  if (!mqtt.connected()) {
    static unsigned long lastReconnectAttempt = 0;
    unsigned long now = millis();
    
    // Only attempt reconnection every 10 seconds
    if (now - lastReconnectAttempt > 10000) {
      lastReconnectAttempt = now;
      if (!mqttReconnect()) {
        return; // Skip publishing if can't connect
      }
    } else {
      return; // Skip if we recently tried to reconnect
    }
  }
  
  // Create JSON payload similar to your working format but with ML data
  char buf[128];
  snprintf(buf, sizeof(buf), 
           "{\"device\":\"ESP32S3_ML\",\"uptime_ms\":%lu,\"svdd\":%.3f,\"mse\":%.4f,\"anomaly\":%s}", 
           millis(), svdd, mse, 
           (svdd > SVDD_THR || mse > MSE_THR) ? "true" : "false");
  
  try {
    if (mqtt.publish(MQTT_TOPIC, buf)) {
      Serial.printf("[MQTT] Published: %s\n", buf);
    } else {
      Serial.println("[MQTT] Publish failed");
    }
  } catch (...) {
    Serial.println("[MQTT] EXCEPTION during publish, disabling MQTT");
    mqtt_enabled = false;
  }
}

/* Network initialization - Using your working setup */
static void netInit()
{
  /* 1️Wi-Fi - Using your credentials */
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print(F("[NET] Wi-Fi connecting"));
  while (WiFi.status() != WL_CONNECTED) { 
    Serial.print('.'); 
    delay(500); 
  }
  Serial.println(F(" ✓"));
  Serial.printf("[NET] WiFi connected! IP: %s\n", WiFi.localIP().toString().c_str());

  /* 2️SNTP (for WG handshakes) */
  configTime(0, 0, "pool.ntp.org", "time.google.com");
  time_t now = 0;
  while (now < 8 * 3600 * 2) { 
    time(&now); 
    delay(500); 
  }

  /* 3️ WireGuard - Using your working configuration */
  if (!wg.begin(WG_LOCAL_IP,
                WG_PRIVATE_KEY,
                WG_ENDPOINT,
                WG_PEER_PUBLIC_KEY,
                WG_PORT))
  {
    Serial.println("[NET] WireGuard bring-up failed");
    while (true) { delay(1000); }
  }
  Serial.printf("[NET] WireGuard up! Tunnel IP: %s\n", WG_LOCAL_IP.toString().c_str());

  /* 4️ MQTT - Using your working broker settings */
  mqtt_enabled = true;  // Start with MQTT enabled
  
  mqtt.setServer(MQTT_BROKER, MQTT_PORT);
  mqtt.setCallback(mqttCallback);
  mqtt.setKeepAlive(30);     // Set keep-alive to 30 seconds
  mqtt.setSocketTimeout(3);  // Set socket timeout to 3 seconds
  
  // Add a delay before attempting MQTT connection
  delay(3000);
  
  // Test basic connectivity to MQTT broker first
  Serial.printf("[MQTT] Testing connectivity to %s:%d...", MQTT_BROKER, MQTT_PORT);
  WiFiClient testClient;
  
  try {
    if (testClient.connect(MQTT_BROKER, MQTT_PORT, 5000)) {  // 5 second timeout
      Serial.println(" OK");
      testClient.stop();
      
      // Now try MQTT connection
      if (mqttReconnect()) {
        Serial.println("[MQTT] Initial connection successful");
      } else {
        Serial.println("[MQTT] Initial connection failed, will retry later");
      }
    } else {
      Serial.println(" FAILED");
      Serial.println("[MQTT] Cannot reach MQTT broker, check WireGuard tunnel");
      Serial.println("[MQTT] Continuing without MQTT...");
      mqtt_enabled = false;
    }
  } catch (...) {
    Serial.println(" EXCEPTION during test");
    mqtt_enabled = false;
  }
}

void setup()
{
  Serial.begin(115200);
  psramInit();
  if (!psramFound()) {
    Serial.println("Enable PSRAM in Tools ▸ OPI PSRAM");
    while (true) {}
  }

  /* I2S mic */
  I2S.setPinsPdmRx(PDM_CLK, PDM_DIN);
  if (!I2S.begin(I2S_MODE_PDM_RX, SAMPLE_RATE,
                 I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO)) {
    Serial.println("I2S.begin failed"); 
    while (true) {}
  }
  Serial.println("Flushing I2S buffers…");
  int16_t tmp[CHUNK];
  for (int i = 0; i < 16; ++i)
    for (int j = 0; j < CHUNK; ++j) tmp[j] = I2S.read();

  /* PSRAM buffers */
  pcm   = (int16_t*)ps_malloc(WIN_SAMPLES * sizeof(int16_t));
  fft   = (float  *)ps_malloc(2 * N_FFT      * sizeof(float));
  hann  = (float  *)ps_malloc(N_FFT          * sizeof(float));
  twid  = (float  *)ps_malloc(N_FFT          * sizeof(float));
  arena = (uint8_t*)ps_malloc(ARENA_BYTES);
  if (!pcm || !fft || !hann || !twid || !arena) {
    Serial.println("RAM allocation failed"); 
    while (true) {}
  }

  for (int i = 0; i < N_FFT; ++i)
    hann[i] = 0.5f - 0.5f * cosf(2 * M_PI * i / (N_FFT - 1));
  dsps_fft2r_init_fc32(twid, N_FFT);

  /* TFL-Micro initialisation */
  if (!([] {
        model = tflite::GetModel(g_model);
        if (model->version() != TFLITE_SCHEMA_VERSION) return false;
        static tflite::MicroMutableOpResolver<8> res;
        res.AddAdd(); res.AddConv2D(); res.AddPad(); res.AddRelu6();
        res.AddReshape(); res.AddStridedSlice(); res.AddTranspose();
        static tflite::MicroInterpreter I(model, res, arena, ARENA_BYTES);
        interp = &I;
        if (interp->AllocateTensors() != kTfLiteOk) return false;
        tinp        = interp->input(0);
        latent_out  = interp->output(0);
        recon_out   = interp->output(1);
        in_scale            = tinp->params.scale;
        in_zp               = tinp->params.zero_point;
        latent_out_scale    = latent_out->params.scale;
        latent_out_zp       = latent_out->params.zero_point;
        recon_out_scale     = recon_out->params.scale;
        recon_out_zp        = recon_out->params.zero_point;
        return true;
      }())) {
    Serial.println("TFLM init failed"); 
    while (true) {}
  }

  Serial.println("DSP + ML ready.");

  /* Networking (Wi-Fi → WG → MQTT) - Using your working setup */
  netInit();
}

void loop()
{
  // Maintain MQTT connection (but protect against crashes)
  if (mqtt_enabled && WiFi.status() == WL_CONNECTED) {
    try {
      mqtt.loop();
    } catch (...) {
      Serial.println("[MQTT] EXCEPTION in mqtt.loop(), disabling MQTT");
      mqtt_enabled = false;
    }
  }
  
  /* 1. Collect audio */
  for (int i = 0; i < CHUNK; ++i) pcm[pcm_idx + i] = I2S.read();
  pcm_idx += CHUNK;
  if (pcm_idx < WIN_SAMPLES) return;

  /* 2. Feature extraction */
  static float mel[MEL_BINS * FRAMES];
  for (int fr = 0, off = 0;
       off <= WIN_SAMPLES - N_FFT;
       off += HOP_LEN, ++fr) {

    for (int i = 0; i < N_FFT; ++i) {
      fft[2 * i    ] = pcm[off + i] * hann[i];
      fft[2 * i + 1] = 0.f;
    }
    dsps_fft2r_fc32(fft, N_FFT);
    dsps_bit_rev_fc32(fft, N_FFT);
    dsps_cplx2reC_fc32(fft, N_FFT);

    static float pwr[N_FFT / 2];
    for (int j = 0; j < N_FFT / 2; ++j) {
      float re = fft[2 * j], im = fft[2 * j + 1];
      pwr[j] = re * re + im * im;
    }

    static float m16[MEL_BINS];
    dspm_mult_f32((const float*)mel_filterbank,
                  pwr, m16, MEL_BINS, N_FFT / 2, 1);

    for (int m = 0; m < MEL_BINS; ++m) {
      float v = (logf(m16[m] + 1e-6f) - MEAN) / STD;
      mel[m * FRAMES + fr] = v;
      int32_t q = lroundf(v / in_scale) + in_zp;
      tinp->data.int8[m * FRAMES + fr] =
          static_cast<int8_t>(std::clamp<int32_t>(q, -128, 127));
    }
  }

  /* 3. Neural network */
  interp->Invoke();

  /* 4. Post-processing (SVDD + MSE) */
  float svdd = 0.f;
  for (int i = 0; i < g_svdd_center_len; ++i) {
    float lv = (latent_out->data.int8[i] - latent_out_zp) * latent_out_scale;
    float d  = lv - g_svdd_center[i];
    svdd += d * d;
  }

  static float recon[MEL_BINS * FRAMES];
  for (int i = 0; i < MEL_BINS * FRAMES; ++i)
    recon[i] = (recon_out->data.int8[i] - recon_out_zp) * recon_out_scale;

  float s = 0.f;
  for (int i = 0; i < MEL_BINS * FRAMES; ++i) {
    float d = mel[i] - recon[i];
    s += d * d;
  }
  float mse_score = s / (MEL_BINS * FRAMES);

#if VERBOSE
  Serial.printf("SVDD %.3f%s  MSE %.4f%s\n",
                svdd,       (svdd       > SVDD_THR ? "*" : ""),
                mse_score,  (mse_score  > MSE_THR  ? "*" : ""));
#endif

  /* 5. Publish - Now using your working MQTT setup through WireGuard tunnel */
  publishScores(svdd, mse_score);

  pcm_idx = 0;
}
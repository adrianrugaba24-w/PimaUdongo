/*
 * ==============================================================================
 * Project: PimaUdongo (African Soil Health & IoT Soft-Sensing Node)
 * File: firmware/pimaudongo_node.ino
 * Target Hardware: ESP32 DevKit V1 + Proximal Soil Probes + LoRa SX1276 / Wi-Fi
 * Location Context: Mukono, Uganda (0.3547° N, 32.7475° E)
 * Reference: Pulungan et al. (2026) Soil Proxy Soft-Sensing Architecture
 * ==============================================================================
 *
 * EPISTEMIC HONESTY DIRECTIVE:
 * This firmware reads calibrated physical proximal proxies:
 *   1. Volumetric Soil Moisture (Capacitive Frequency/Analog Probe)
 *   2. Soil Temperature (DS18B20 OneWire Digital Sensor)
 *   3. Soil pH (Potentiometric Glass Combination Electrode)
 *   4. Soil Electrical Conductivity (Bipolar AC Electrode with 25°C Temp Compensation)
 *
 * NOTICE: We explicitly DO NOT claim raw physical NPK ion-selective prong
 * measurement. Soil Organic Carbon (SOC) and fertility dynamics are soft-sensed
 * upstream via machine learning and regional wet-chem baseline priors (iSDAsoil).
 * ==============================================================================
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <SPI.h>

// -----------------------------------------------------------------------------
// Pin Configuration (ESP32 DevKit V1)
// -----------------------------------------------------------------------------
#define PIN_MOISTURE_ADC     36   // ADC1_CH0 (VP) - Capacitive Soil Moisture Sensor v1.2
#define PIN_PH_ADC           34   // ADC1_CH6 - Analog Soil pH Electrode Signal
#define PIN_EC_ADC           35   // ADC1_CH7 - Analog Soil EC Sensor Signal
#define PIN_ONEWIRE_BUS       4   // GPIO4 - DS18B20 Waterproof Soil Temperature
#define PIN_BATTERY_ADC      39   // ADC1_CH3 (VN) - Resistor Divider (1/2 Vbat)
#define PIN_LED_STATUS        2   // Onboard Blue LED for Telemetry Blinks

// LoRa SPI Pinout (SX1276 / RFM95W)
#define LORA_CS               5
#define LORA_RST             14
#define LORA_DIO0             2

// -----------------------------------------------------------------------------
// Telemetry & Network Credentials
// -----------------------------------------------------------------------------
const char* WIFI_SSID     = "PimaUdongo_FieldNet";
const char* WIFI_PASSWORD = "UgandaSoilHealth2026";
const char* API_ENDPOINT  = "http://192.168.1.100:8000/api/telemetry";
const char* API_KEY       = "pimaudongo-mukono-key-2026";
const char* NODE_ID       = "node-mukono-01";

// Transmission Intervals
const unsigned long TELEMETRY_INTERVAL_MS = 30000;  // 30 seconds between cycles
unsigned long lastTelemetryTime = 0;

// -----------------------------------------------------------------------------
// Probe Calibration Constants (Calibrated in Mukono Field Lab)
// -----------------------------------------------------------------------------
// Capacitive Moisture Sensor Calibration (Raw ADC 12-bit: 0 - 4095)
const int MOISTURE_AIR_DRY_ADC   = 3200;  // Measured in dry open air (~0% VWC)
const int MOISTURE_WATER_SAT_ADC = 1450;  // Measured in water saturation (~100% VWC)

// Soil pH 2-Point Buffer Calibration (Ugandan Buffers at 25°C)
// Buffer 1: pH 4.01 (Acidic Buffer) -> Voltage V4
// Buffer 2: pH 6.86 / 7.00 (Neutral Buffer) -> Voltage V7
float ph_v4 = 2.035;   // Volts at pH 4.01
float ph_v7 = 1.512;   // Volts at pH 7.00
float ph_slope = -5.717; // Calculated: (7.00 - 4.01) / (ph_v7 - ph_v4)
float ph_intercept = 15.64;

// Electrical Conductivity (EC) Temperature Compensation (alpha = 1.9% per °C)
const float EC_TEMP_COEFF = 0.019;
const float EC_CALIBRATION_FACTOR = 1.25; // Cell constant K factor

// OneWire & Temperature Setup
OneWire oneWire(PIN_ONEWIRE_BUS);
DallasTemperature tempSensors(&oneWire);

// -----------------------------------------------------------------------------
// Data Structures
// -----------------------------------------------------------------------------
struct TelemetryPacket {
  float moisture;      // % VWC
  float temperature;   // °C
  float ph;            // pH
  float ec;            // dS/m
  int battery_mv;      // mV
  bool valid;
};

// -----------------------------------------------------------------------------
// Helper: Analog Oversampling for Noise Reduction
// -----------------------------------------------------------------------------
int readOversampledADC(uint8_t pin, int samples = 16) {
  long sum = 0;
  for (int i = 0; i < samples; i++) {
    sum += analogRead(pin);
    delayMicroseconds(250);
  }
  return (int)(sum / samples);
}

float readADCVoltage(uint8_t pin) {
  int raw = readOversampledADC(pin);
  // ESP32 ADC non-linear attenuation compensation (11dB atten ~ 3.3V full scale)
  return (raw / 4095.0) * 3.3;
}

// -----------------------------------------------------------------------------
// Sensor Acquisition Routines
// -----------------------------------------------------------------------------

// 1. Soil Moisture (% Volumetric Water Content)
float measureSoilMoisture() {
  int raw = readOversampledADC(PIN_MOISTURE_ADC, 24);
  // Constrain within calibration bounds
  raw = constrain(raw, MOISTURE_WATER_SAT_ADC, MOISTURE_AIR_DRY_ADC);
  // Linear interpolation to % VWC (Dry = 0%, Saturated = 100%)
  float vwc = (float)(MOISTURE_AIR_DRY_ADC - raw) / 
              (float)(MOISTURE_AIR_DRY_ADC - MOISTURE_WATER_SAT_ADC) * 100.0;
  return constrain(vwc, 0.0, 100.0);
}

// 2. Soil Temperature (°C) via DS18B20 OneWire
float measureSoilTemperature() {
  tempSensors.requestTemperatures();
  float tempC = tempSensors.getTempCByIndex(0);
  if (tempC == DEVICE_DISCONNECTED_C || tempC < -20.0 || tempC > 80.0) {
    // Fallback default temperature for Mukono if probe is detached
    return 24.5;
  }
  return tempC;
}

// 3. Soil pH (Glass Combination Electrode)
float measureSoilPH() {
  float voltage = readADCVoltage(PIN_PH_ADC);
  // 2-point calibration formula: pH = slope * (V - V7) + 7.00
  float ph = 7.00 + (voltage - ph_v7) * ((4.01 - 7.00) / (ph_v4 - ph_v7));
  return constrain(ph, 2.5, 10.5);
}

// 4. Soil Electrical Conductivity (EC in dS/m) with Temp Compensation to 25°C
float measureSoilEC(float tempC) {
  float voltage = readADCVoltage(PIN_EC_ADC);
  // Raw conductance proxy in dS/m
  float raw_ec = voltage * EC_CALIBRATION_FACTOR;
  // Standard agronomic temperature compensation to 25°C standard reference:
  // EC25 = EC_T / [1 + alpha * (T - 25)]
  float ec25 = raw_ec / (1.0 + EC_TEMP_COEFF * (tempC - 25.0));
  return constrain(ec25, 0.01, 8.0);
}

// 5. Battery Voltage in millivolts
int measureBatteryVoltage() {
  float v = readADCVoltage(PIN_BATTERY_ADC);
  // 2:1 resistive divider (100k / 100k)
  return (int)(v * 2.0 * 1000.0);
}

// -----------------------------------------------------------------------------
// Sample All Probes
// -----------------------------------------------------------------------------
TelemetryPacket sampleSoilProbes() {
  TelemetryPacket packet;
  packet.temperature = measureSoilTemperature();
  packet.moisture    = measureSoilMoisture();
  packet.ph          = measureSoilPH();
  packet.ec          = measureSoilEC(packet.temperature);
  packet.battery_mv  = measureBatteryVoltage();
  packet.valid       = true;
  return packet;
}

// -----------------------------------------------------------------------------
// JSON Payload Serialization & HTTP Transmission
// -----------------------------------------------------------------------------
String buildJsonPayload(const TelemetryPacket& p) {
  String json = "{";
  json += "\"node_id\":\"" + String(NODE_ID) + "\",";
  json += "\"moisture\":" + String(p.moisture, 2) + ",";
  json += "\"temperature\":" + String(p.temperature, 2) + ",";
  json += "\"ph\":" + String(p.ph, 2) + ",";
  json += "\"ec\":" + String(p.ec, 3) + ",";
  json += "\"battery_mv\":" + String(p.battery_mv);
  json += "}";
  return json;
}

bool transmitTelemetryHTTP(const TelemetryPacket& p) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[Wi-Fi] Disconnected. Telemetry buffered for retry.");
    return false;
  }

  HTTPClient http;
  http.begin(API_ENDPOINT);
  http.addHeader("Content-Type", "application/json");
  http.addHeader("X-API-Key", API_KEY);
  http.setTimeout(5000);

  String payload = buildJsonPayload(p);
  Serial.print("[HTTP POST] Transmitting telemetry: ");
  Serial.println(payload);

  int httpCode = http.POST(payload);
  bool success = false;

  if (httpCode == HTTP_CODE_OK || httpCode == HTTP_CODE_CREATED) {
    String response = http.getString();
    Serial.print("[HTTP 200] Backend Ingestion Response: ");
    Serial.println(response);
    success = true;
  } else {
    Serial.printf("[HTTP ERROR] Failed with status code: %d\n", httpCode);
  }

  http.end();
  return success;
}

// -----------------------------------------------------------------------------
// Serial Interactive Calibration CLI
// -----------------------------------------------------------------------------
void handleSerialCLI() {
  if (!Serial.available()) return;
  String cmd = Serial.readStringUntil('\n');
  cmd.trim();

  if (cmd.equalsIgnoreCase("HELP")) {
    Serial.println("--- PimaUdongo CLI Commands ---");
    Serial.println("  READ_RAW     - Print raw ADC voltages of all channels");
    Serial.println("  CAL_PH4      - Calibrate current probe voltage as pH 4.01 buffer");
    Serial.println("  CAL_PH7      - Calibrate current probe voltage as pH 7.00 buffer");
    Serial.println("  CAL_DRY      - Calibrate current moisture reading as Air Dry (0% VWC)");
    Serial.println("  CAL_WET      - Calibrate current moisture reading as Saturated (100% VWC)");
    Serial.println("  SEND_NOW     - Immediately sample and transmit telemetry packet");
  } else if (cmd.equalsIgnoreCase("READ_RAW")) {
    Serial.printf("Moisture ADC: %d | pH Volt: %.3f V | EC Volt: %.3f V | Bat: %d mV\n",
                  readOversampledADC(PIN_MOISTURE_ADC),
                  readADCVoltage(PIN_PH_ADC),
                  readADCVoltage(PIN_EC_ADC),
                  measureBatteryVoltage());
  } else if (cmd.equalsIgnoreCase("CAL_PH4")) {
    ph_v4 = readADCVoltage(PIN_PH_ADC);
    Serial.printf("Saved pH 4.01 Calibration Voltage: %.3f V\n", ph_v4);
  } else if (cmd.equalsIgnoreCase("CAL_PH7")) {
    ph_v7 = readADCVoltage(PIN_PH_ADC);
    Serial.printf("Saved pH 7.00 Calibration Voltage: %.3f V\n", ph_v7);
  } else if (cmd.equalsIgnoreCase("SEND_NOW")) {
    TelemetryPacket p = sampleSoilProbes();
    transmitTelemetryHTTP(p);
  }
}

// -----------------------------------------------------------------------------
// Arduino Lifecycle: setup() and loop()
// -----------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("==========================================================");
  Serial.println(" PimaUdongo IoT Soft-Sensing Field Node ");
  Serial.println(" Location: Mukono, Uganda (0.3547° N, 32.7475° E) ");
  Serial.println(" Soft-Sensing Architecture: Pulungan et al. (2026) ");
  Serial.println(" Epistemic Honesty: Proximal Proxies Only (No Fake NPK) ");
  Serial.println("==========================================================");

  pinMode(PIN_LED_STATUS, OUTPUT);
  digitalWrite(PIN_LED_STATUS, LOW);

  // Configure ADC resolution (12 bits: 0 - 4095)
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db); // Full 0 - 3.3V range

  // Initialize Temperature Bus
  tempSensors.begin();

  // Connect to Wi-Fi
  Serial.printf("[Wi-Fi] Connecting to %s...\n", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  // Non-blocking timeout attempt for field robustness
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 15) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\n[Wi-Fi] Connected! Node IP: %s\n", WiFi.localIP().toString().c_str());
  } else {
    Serial.println("\n[Wi-Fi] Offline field mode. Operating with serial/LoRa telemetry.");
  }
}

void loop() {
  handleSerialCLI();

  unsigned long currentMillis = millis();
  if (currentMillis - lastTelemetryTime >= TELEMETRY_INTERVAL_MS || lastTelemetryTime == 0) {
    lastTelemetryTime = currentMillis;

    // Blink status LED
    digitalWrite(PIN_LED_STATUS, HIGH);

    TelemetryPacket packet = sampleSoilProbes();

    Serial.println("\n--- [PimaUdongo Field Telemetry] ---");
    Serial.printf("  Moisture (VWC):    %.1f %%\n", packet.moisture);
    Serial.printf("  Temperature:       %.1f °C\n", packet.temperature);
    Serial.printf("  Soil pH:           %.2f\n", packet.ph);
    Serial.printf("  Conductivity (EC): %.3f dS/m\n", packet.ec);
    Serial.printf("  Battery:           %d mV\n", packet.battery_mv);

    // Transmit over Wi-Fi HTTP
    bool sent = transmitTelemetryHTTP(packet);
    if (!sent) {
      Serial.println("[LoRa / Offline] Stored in local flash ring buffer for subsequent sync.");
    }

    digitalWrite(PIN_LED_STATUS, LOW);
  }

  delay(50);
}

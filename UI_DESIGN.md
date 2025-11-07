# 📱 EdgeSonic Android UI Design

## Overview

Clean, mobile-optimized UI with **tab-based navigation** and **Material Design 3**.

---

## 🎨 Design Principles

- **Tab Navigation** - Easy access to Live, File, and MQTT features
- **Color Coding** - Green (normal), Red (anomaly), Grey (stopped)
- **Large Touch Targets** - Buttons optimized for mobile
- **Visual Feedback** - Icons, colors, and animations
- **Clean Layout** - Cards with rounded corners and subtle borders

---

## 📱 Screen Layouts

### App Bar
```
┌────────────────────────────────────┐
│  EdgeSonic                    ✓/✗  │  ← Model status indicator
├────────────────────────────────────┤
│  🎤 Live  │  📄 File  │  🔌 MQTT   │  ← Tabs
└────────────────────────────────────┘
```

---

### 1️⃣ **Live Tab** (Real-time Inference)

#### When Stopped:
```
┌────────────────────────────────────┐
│        [Grey Mic Icon - 64px]      │
│                                    │
│            STOPPED                 │
│         (grey color)               │
│                                    │
└────────────────────────────────────┘

┌────────────────────────────────────┐
│  [START LIVE CAPTURE - Full width] │
└────────────────────────────────────┘

┌────────────────────────────────────┐
│  ℹ️ How it works                   │
│                                    │
│  • Captures audio at 16kHz         │
│  • Analyzes 128-frame windows      │
│  • Detects anomalies in real-time  │
│  • 5-15ms latency                  │
└────────────────────────────────────┘
```

#### When Running (Normal):
```
┌────────────────────────────────────┐
│   [Green Mic Icon - 64px]          │
│                                    │
│          LISTENING                 │
│        ✓ Normal                    │
│      (green color)                 │
└────────────────────────────────────┘

┌─────────────────┬──────────────────┐
│  📊 Chunks      │  📈 RMS          │
│     1,234       │    0.042         │
│   (blue icon)   │ (purple icon)    │
└─────────────────┴──────────────────┘

┌─────────────────┬──────────────────┐
│  🎯 Score       │  ⚡ Latency      │
│    0.0034       │    8.2ms         │
│  (green icon)   │ (orange icon)    │
└─────────────────┴──────────────────┘

┌────────────────────────────────────┐
│  [STOP CAPTURE - Red, Full width]  │
└────────────────────────────────────┘
```

#### When Running (Anomaly Detected):
```
┌────────────────────────────────────┐
│    [Red Mic Icon - 64px]           │
│                                    │
│          LISTENING                 │
│    ⚠️ ANOMALY DETECTED             │
│       (red color)                  │
│  [Red background card]             │
└────────────────────────────────────┘

┌─────────────────┬──────────────────┐
│  📊 Chunks      │  📈 RMS          │
│     1,234       │    0.156         │
└─────────────────┴──────────────────┘

┌─────────────────┬──────────────────┐
│  🎯 Score       │  ⚡ Latency      │
│    0.0189       │    12.5ms        │
│  (RED - HIGH!)  │                  │
└─────────────────┴──────────────────┘
```

---

### 2️⃣ **File Tab** (Audio Upload)

```
┌────────────────────────────────────┐
│  🎵 Audio File Processing          │
│                                    │
│  ┌──────────────────────────────┐ │
│  │ 🎵 my_audio_file.wav         │ │
│  └──────────────────────────────┘ │
│                                    │
│  [SELECT AUDIO FILE - Blue]        │
│                                    │
│  [PROCESS - Outlined, Coming Soon] │
└────────────────────────────────────┘

┌────────────────────────────────────┐
│  ℹ️ File Processing                │
│                                    │
│  Upload audio files to analyze     │
│  for anomalies offline. Results    │
│  will be displayed with timestamps │
│  and exportable to CSV.            │
└────────────────────────────────────┘
```

---

### 3️⃣ **MQTT Tab** (Connectivity)

```
┌────────────────────────────────────┐
│  🔌 MQTT Integration               │
│                                    │
│  Connect to your MQTT broker to    │
│  receive telemetry or simulate     │
│  ESP32 device payloads.            │
│                                    │
│  [MQTT CONNECTION TEST - Blue]     │
│                                    │
│  [ESP32 SIMULATOR - Outlined]      │
└────────────────────────────────────┘

┌────────────────────────────────────┐
│  ✅ Features                       │
│                                    │
│  ✓ Connect to any MQTT broker      │
│  ✓ Subscribe to topics             │
│  ✓ Publish anomaly results         │
│  ✓ Simulate ESP32 telemetry        │
│  ✓ Real-time message monitoring    │
└────────────────────────────────────┘
```

---

## 🎨 Color Palette

### Primary Colors
- **Teal** (#009688) - Primary brand color
- **Light Teal** - Cards and accents

### Status Colors
- **Green** (#4CAF50) - Normal operation
- **Red** (#F44336) - Anomaly detected / Error
- **Grey** (#9E9E9E) - Stopped / Inactive
- **Blue** (#2196F3) - Info / Metrics
- **Orange** (#FF9800) - Warning / Latency
- **Purple** (#9C27B0) - Audio metrics

### Background Colors
- **White** - Main background
- **Light Grey** (#F5F5F5) - Card borders
- **Green Tint** (#E8F5E9) - Normal status cards
- **Red Tint** (#FFEBEE) - Anomaly status cards
- **Blue Tint** (#E3F2FD) - Info cards

---

## 📐 Measurements

### Spacing
- Card padding: 16-20px
- Button padding: 14-16px vertical
- Icon size (status): 64px
- Icon size (metrics): 28px
- Gap between cards: 16px

### Typography
- Headline: 24px, Bold
- Title: 18-20px, Bold
- Body: 14-16px, Regular
- Metric Values: 20px, Bold
- Metric Labels: 12px, Regular

### Borders
- Card border radius: 16px
- Container radius: 8px
- Border width: 1px

---

## 🔄 States & Animations

### Live Inference States
1. **Stopped** - Grey mic, "STOPPED" text
2. **Running (Normal)** - Green mic, "LISTENING", ✓ Normal
3. **Running (Anomaly)** - Red mic, "LISTENING", ⚠️ ANOMALY

### Button States
1. **Start** - Teal, play icon
2. **Stop** - Red, stop icon
3. **Disabled** - Grey, no interaction

### Card Backgrounds
- **Default** - White with grey border
- **Active Normal** - Green tint (#E8F5E9)
- **Active Anomaly** - Red tint (#FFEBEE)
- **Info** - Blue tint (#E3F2FD)

---

## 📱 Responsive Design

### Small Screens (< 360dp)
- Single column metric cards
- Compact padding
- Smaller fonts

### Medium Screens (360-480dp)
- 2-column metric grid
- Standard padding
- Default fonts

### Large Screens (> 480dp)
- 2-column metric grid
- Generous padding
- Larger fonts

---

## ✨ Key Features

### Visual Feedback
- ✅ Color-coded status indicators
- ✅ Icon-based metrics
- ✅ Real-time value updates
- ✅ Clear error messages
- ✅ Loading states

### Touch Optimization
- ✅ Large 48dp+ touch targets
- ✅ Clear button labels
- ✅ Swipe between tabs
- ✅ Scroll for content

### Accessibility
- ✅ High contrast colors
- ✅ Clear icons and labels
- ✅ Status in text and color
- ✅ Readable font sizes

---

## 🚀 Usage Flow

### Live Inference Flow
```
1. User opens app
   ↓
2. Sees "Model Ready" in app bar (green checkmark)
   ↓
3. Navigates to "Live" tab (default)
   ↓
4. Reads "How it works" info card
   ↓
5. Taps "START LIVE CAPTURE"
   ↓
6. Grants microphone permission
   ↓
7. Sees status change to "LISTENING" (green)
   ↓
8. Watches metrics update in real-time
   ↓
9. Sees anomaly detection if triggered (red)
   ↓
10. Taps "STOP CAPTURE" when done
```

### File Processing Flow
```
1. User navigates to "File" tab
   ↓
2. Taps "SELECT AUDIO FILE"
   ↓
3. Picks file from device
   ↓
4. Sees file name displayed
   ↓
5. Taps "PROCESS" (coming soon)
   ↓
6. Views results with timestamps
```

### MQTT Flow
```
1. User navigates to "MQTT" tab
   ↓
2. Reads features list
   ↓
3. Taps "MQTT CONNECTION TEST"
   ↓
4. Configures broker settings
   ↓
5. Connects and subscribes
   ↓
OR
   ↓
3. Taps "ESP32 SIMULATOR"
   ↓
4. Simulates device telemetry
```

---

## 🎯 Design Goals Achieved

✅ **Mobile-First** - Optimized for Android screens
✅ **Clear Status** - Instant visual feedback
✅ **Easy Navigation** - Tab-based, one tap away
✅ **Visual Hierarchy** - Important info stands out
✅ **Touch-Friendly** - Large buttons and targets
✅ **Informative** - Info cards explain features
✅ **Professional** - Clean Material Design 3
✅ **Accessible** - High contrast, clear text

---

**Built for real-time anomaly detection on the edge! 🚀**

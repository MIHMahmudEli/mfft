/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: "class",
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // True classic dark: obsidian & deep pitch black for premium OLED UX
        dark: {
          bg: "#000000",
          surface: "#09090b",
          card: "#121215",
          elevated: "#18181b",
          border: "#27272a",
          "border-subtle": "#18181b",
        },
        primary: {
          50: "#eff6ff",
          100: "#dbeafe",
          200: "#bfdbfe",
          300: "#93c5fd",
          400: "#60a5fa",
          500: "#3b82f6",
          600: "#2563eb",
          700: "#1d4ed8",
          800: "#1e40af",
          900: "#1e3a8a",
        },
      },
      boxShadow: {
        "glow-sm": "0 0 20px -5px rgba(59, 130, 246, 0.4)",
        glow: "0 0 40px -10px rgba(59, 130, 246, 0.5)",
        "glow-emerald": "0 0 30px -8px rgba(16, 185, 129, 0.45)",
        "glow-rose": "0 0 30px -8px rgba(244, 63, 94, 0.45)",
        "inner-glow": "inset 0 0 20px 0 rgba(255, 255, 255, 0.05)",
      },
      animation: {
        "fade-in": "fadeIn 0.4s cubic-bezier(0.16, 1, 0.3, 1)",
        "slide-up": "slideUp 0.5s cubic-bezier(0.16, 1, 0.3, 1)",
        shimmer: "shimmer 6s linear infinite",
        float: "float 6s ease-in-out infinite",
        scan: "scan 2s ease-in-out infinite",
        pulse_slow: "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        slideUp: {
          "0%": { opacity: "0", transform: "translateY(16px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        shimmer: {
          "0%, 100%": { backgroundPosition: "0% 50%" },
          "50%": { backgroundPosition: "100% 50%" },
        },
        float: {
          "0%, 100%": { transform: "translateY(0)" },
          "50%": { transform: "translateY(-6px)" },
        },
        scan: {
          "0%": { top: "0%" },
          "50%": { top: "96%" },
          "100%": { top: "0%" },
        },
      },
    },
  },
  plugins: [],
};

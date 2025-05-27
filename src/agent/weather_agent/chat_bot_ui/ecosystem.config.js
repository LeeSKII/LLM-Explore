module.exports = {
  apps: [
    {
      name: "weather-agent-python-app",
      script: "streamlit",
      interpreter: "none",
      args: "run weather_chatbot.py",
      autorestart: true,
      env: {
        ENVIRONMENT: "production",
        PORT: 8501,
      },
      watch: true,
      error_file: "./logs/error.log",
      out_file: "./logs/out.log",
      log_date_format: "YYYY-MM-DD HH:mm:ss",
    },
  ],
};

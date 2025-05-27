module.exports = {
  apps: [
    {
      name: "chainlit-chat-python-app",
      script: "home.py",
      interpreter: "python",
      args: "--port 8000 --headless",
      env: {
        ENVIRONMENT: "production",
      },
      watch: true,
      error_file: "./logs/error.log",
      out_file: "./logs/out.log",
      log_date_format: "YYYY-MM-DD HH:mm:ss",
    },
  ],
};

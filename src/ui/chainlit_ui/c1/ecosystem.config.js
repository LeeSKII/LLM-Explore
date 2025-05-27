module.exports = {
  apps: [
    {
      name: "chainlit-chat-python-app",
      script: "chainlit",
      interpreter: "none",
      args: "run home.py --host 0.0.0.0 --port 8500 --headless",
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

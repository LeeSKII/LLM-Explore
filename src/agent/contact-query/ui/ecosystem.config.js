module.exports = {
  apps: [
    {
      name: "chainlit-contact-python-app",
      script: "chainlit",
      interpreter: "none",
      args: "run chainlit-ui.py --host 0.0.0.0 --port 8502 --headless",
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

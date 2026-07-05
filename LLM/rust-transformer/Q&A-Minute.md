# Q&A Minute - Git Push Reference Already Exists

## Entry: 2026-07-05

* **Summary**: The Git error `reference already exists` is caused by a leftover reference lock file (`main.lock`) that prevents Git from updating the remote branch reference locally.
* **Issue**: A crashed or interrupted Git process left a lock file at `.git/refs/remotes/origin/main.lock`.
* **Approach**: Remove the stale lock file manually, then run `git fetch` and `git push` again.
* **Example or Analogy**: It is like a "Do Not Disturb" sign left on a hotel room door by mistake—even though the guests have checked out, no one else can enter until you remove the sign.

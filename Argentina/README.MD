## Data Leakage Fixes

Were identified and corrected three main sources of potential data leakage to ensure our model generalizes to unseen data.

---

### 1. Duplicate Records

* **Issue:** Some rows were duplicated in the dataset.
* **Fix:** Removed all duplicate rows before splitting the dataset.
* **Reason:** Duplicates can artificially inflate accuracy because the model might encounter the same record in both the training and test sets, which it has already "memorized."

---

### 2. Scaler Fitting on Entire Dataset

* **Issue:** The `StandardScaler` was initially fit on the full dataset before the train/test split.
* **Fix:** The `StandardScaler` is now fit **only** on the training data and then used to transform both the training and test sets.
* **Reason:** Fitting the scaler on the full dataset leaks statistical information (like the mean and standard deviation) from the test set into the training process. This gives the model an unrealistic advantage and leads to overly optimistic performance metrics.

---

### 3. Audit ID Feature

* **Issue:** The `audit_id` column had a direct correlation with the target variable (`loan_defaulted`). For example, `audit_id = 0` always corresponded to `loan_defaulted = 1`.
* **Fix:** Removed the `audit_id` column entirely from the features.
* **Reason:** A feature that directly encodes the target is a classic example of **target leakage**. It allows the model to "cheat" by learning a simple rule instead of learning the actual underlying patterns from the other features.
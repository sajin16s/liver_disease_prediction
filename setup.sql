CREATE DATABASE IF NOT EXISTS liver_db CHARACTER SET utf8mb4;
USE liver_db;

CREATE TABLE IF NOT EXISTS patients (
    patient_id   INT AUTO_INCREMENT PRIMARY KEY,
    patient_name VARCHAR(100) NOT NULL,
    age          TINYINT UNSIGNED NOT NULL,
    gender       ENUM('Male','Female') NOT NULL,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_patient_name (patient_name)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id              INT AUTO_INCREMENT PRIMARY KEY,
    patient_id                 INT NOT NULL,
    total_bilirubin            FLOAT NOT NULL,
    direct_bilirubin           FLOAT NOT NULL,
    alkaline_phosphotase       FLOAT NOT NULL,
    alamine_aminotransferase   FLOAT NOT NULL,
    aspartate_aminotransferase FLOAT NOT NULL,
    total_proteins             FLOAT NOT NULL,
    albumin                    FLOAT NOT NULL,
    albumin_globulin_ratio     FLOAT NOT NULL,
    probability                FLOAT NOT NULL,
    risk_level                 ENUM('HIGH','LOW') NOT NULL,
    threshold_used             FLOAT NOT NULL,
    predicted_at               TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_patient FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE,
    INDEX idx_predicted_at (predicted_at)
) ENGINE=InnoDB;



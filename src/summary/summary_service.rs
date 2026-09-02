use std::path::PathBuf;

pub struct SummaryService {
    model_name: String,
    prompt_path: PathBuf,
    output_dir: PathBuf,
}

impl SummaryService {
    pub fn new() -> Self {
        Self {
            model_name: "gemma4:e4b".to_string(),
            prompt_path: PathBuf::from("./src/transcriptor.md"),
            output_dir: PathBuf::from("./outputs"),
        }
    }

    pub fn with_model(mut self, model_name: impl Into<String>) -> Self {
        self.model_name = model_name.into();
        self
    }

    pub fn with_prompt_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.prompt_path = path.into();
        self
    }

    pub fn with_output_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.output_dir = dir.into();
        self
    }
}

impl Default for SummaryService {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_initialization() {
        // Arrange

        // Act
        let service = SummaryService::new();

        // Assert
        assert_eq!(service.model_name, "gemma4:e4b");
        assert_eq!(service.prompt_path, PathBuf::from("./src/transcriptor.md"));
        assert_eq!(service.output_dir, PathBuf::from("./outputs"));
    }

    #[test]
    fn test_default_trait_implementation() {
        // Arrange

        // Act
        let service = SummaryService::default();

        // Assert
        assert_eq!(service.model_name, "gemma4:e4b");
        assert_eq!(service.prompt_path, PathBuf::from("./src/transcriptor.md"));
        assert_eq!(service.output_dir, PathBuf::from("./outputs"));
    }

    #[test]
    fn test_builder_with_model() {
        // Arrange
        let service = SummaryService::new();
        let expected_model = "llama3.1:latest";

        // Act
        let updated_service = service.with_model(expected_model);

        // Assert
        assert_eq!(updated_service.model_name, expected_model);
        assert_eq!(updated_service.prompt_path, PathBuf::from("./src/transcriptor.md"));
        assert_eq!(updated_service.output_dir, PathBuf::from("./outputs"));
    }

    #[test]
    fn test_builder_with_prompt_path() {
        // Arrange
        let service = SummaryService::new();
        let custom_path = PathBuf::from("/custom/path/prompt.md");

        // Act
        let updated_service = service.with_prompt_path(custom_path.clone());

        // Assert
        assert_eq!(updated_service.prompt_path, custom_path);
    }

    #[test]
    fn test_builder_with_output_dir() {
        // Arrange
        let service = SummaryService::new();
        let custom_dir = PathBuf::from("/tmp/custom_outputs");

        // Act
        let updated_service = service.with_output_dir(custom_dir.clone());

        // Assert
        assert_eq!(updated_service.output_dir, custom_dir);
    }

    #[test]
    fn test_builder_chaining() {
        // Arrange
        let initial_service = SummaryService::new();
        let target_model = "mistral:7b";
        let target_prompt = "./prompts/custom.md";
        let target_output = "./custom_out";

        // Act
        let configured_service = initial_service
            .with_model(target_model)
            .with_prompt_path(target_prompt)
            .with_output_dir(target_output);

        // Assert
        assert_eq!(configured_service.model_name, target_model);
        assert_eq!(configured_service.prompt_path, PathBuf::from(target_prompt));
        assert_eq!(configured_service.output_dir, PathBuf::from(target_output));
    }
}
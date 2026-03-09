use nekowg::{Action, actions};
use nekowg_macros::register_action;
use schemars::JsonSchema;
use serde::Deserialize;

#[test]
fn test_action_macros() {
    actions!(
        test_only,
        [
            SomeAction,
            /// Documented action
            SomeActionWithDocs,
        ]
    );

    #[derive(PartialEq, Clone, Deserialize, JsonSchema, Action)]
    #[action(namespace = test_only)]
    #[serde(deny_unknown_fields)]
    struct AnotherAction;

    #[derive(PartialEq, Clone, nekowg::private::serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct RegisterableAction {}

    register_action!(RegisterableAction);

    impl nekowg::Action for RegisterableAction {
        fn boxed_clone(&self) -> Box<dyn nekowg::Action> {
            unimplemented!()
        }

        fn partial_eq(&self, _action: &dyn nekowg::Action) -> bool {
            unimplemented!()
        }

        fn name(&self) -> &'static str {
            unimplemented!()
        }

        fn name_for_type() -> &'static str
        where
            Self: Sized,
        {
            unimplemented!()
        }

        fn build(_value: serde_json::Value) -> anyhow::Result<Box<dyn nekowg::Action>>
        where
            Self: Sized,
        {
            unimplemented!()
        }
    }
}
